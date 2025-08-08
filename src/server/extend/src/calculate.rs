use ahash::RandomState;
use ahash::{AHashMap, AHashSet};
use jieba_rs::Jieba;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rapidfuzz::distance::levenshtein;
use regex::Regex;
use std::hash::{BuildHasher, Hash, Hasher};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimType {
    Levenshtein,
    Jaccard,
    Simhash,
}

#[pyclass]
pub struct TextSimilarity {
    method: SimType,
    jieba: Jieba,
    remove_brackets_re: Regex,
    non_word_re: Regex,
    leading_num_re: Regex,
    stopwords: Vec<String>,
    hasher_builder: RandomState,
}

#[pymethods]
impl TextSimilarity {
    #[new]
    fn new(method: &str) -> PyResult<Self> {
        let method = match method.to_lowercase().as_str() {
            "levenshtein" => SimType::Levenshtein,
            "jaccard" => SimType::Jaccard,
            "simhash" => SimType::Simhash,
            _ => return Err(PyValueError::new_err("Similarity method")),
        };

        Ok(TextSimilarity {
            method,
            jieba: Jieba::new(),
            remove_brackets_re: Regex::new(r"[(\[（【].*?[)\]）】]").unwrap(),
            non_word_re: Regex::new(r"[^\p{Script=Han}\p{L}\p{N}]+").unwrap(),
            leading_num_re: Regex::new(r"^(?:no\.?|NO\.?\s*)?[#\-]*\d+[#\-]*").unwrap(),
            stopwords: vec![
                "的".to_string(),
                "号".to_string(),
                "编号".to_string(),
                "unit".to_string(),
            ],
            // 固定种子保证哈希稳定性
            hasher_builder: RandomState::with_seeds(1, 2, 3, 4),
        })
    }

    fn calculate(&self, s1: &str, s2: &str) -> PyResult<f64> {
        let n1 = self.normalize(s1);
        let n2 = self.normalize(s2);

        if n1.is_empty() && n2.is_empty() {
            return Ok(1.0);
        }

        // 分词缓存
        let token_vec1 = self.tokenize(&n1);
        let token_vec2 = self.tokenize(&n2);

        let token_sim = match self.method {
            SimType::Levenshtein => self.levenshtein_similarity(&n1, &n2),
            SimType::Jaccard => self.jaccard_similarity_tokens(&token_vec1, &token_vec2),
            SimType::Simhash => self.simhash_similarity_tokens(&token_vec1, &token_vec2),
        };
        let char_sim = match self.method {
            SimType::Levenshtein => token_sim,
            SimType::Jaccard => self.jaccard_similarity_chars(&n1, &n2),
            SimType::Simhash => self.simhash_similarity_chars(&n1, &n2),
        };

        Ok(token_sim.max(char_sim))
    }

    fn batch_calculate(&self, pairs: Vec<(String, String)>) -> PyResult<Vec<f64>> {
        let mut results = Vec::with_capacity(pairs.len());
        for (a, b) in pairs.into_iter() {
            results.push(self.calculate(&a, &b)?);
        }
        Ok(results)
    }
}

impl TextSimilarity {
    fn normalize(&self, s: &str) -> String {
        let mut t = s.nfkc().collect::<String>();
        t = self.remove_brackets_re.replace_all(&t, "").into_owned();
        t = self.leading_num_re.replace(&t, "").into_owned();
        t = self.non_word_re.replace_all(&t, " ").into_owned();

        // collapse spaces and trim first, 再做停用词去除会更准
        let collapsed = t.split_whitespace().collect::<Vec<&str>>().join(" ");

        // 利用分词过滤停用词，避免多次replace
        let tokens = self.tokenize(&collapsed);

        tokens.join(" ").trim().to_string()
    }

    fn levenshtein_similarity(&self, s1: &str, s2: &str) -> f64 {
        let dist = levenshtein::distance(s1.chars(), s2.chars());
        let max_len = s1.chars().count().max(s2.chars().count()) as f64;
        if max_len == 0.0 {
            1.0
        } else {
            1.0 - (dist as f64 / max_len)
        }
    }

    fn tokenize(&self, s: &str) -> Vec<String> {
        if s.is_empty() {
            return Vec::new();
        }
        self.jieba
            .cut(s, false)
            .into_iter()
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty() && !self.stopwords.contains(x))
            .collect()
    }

    fn jaccard_similarity_tokens(&self, tokens1: &[String], tokens2: &[String]) -> f64 {
        if tokens1.is_empty() && tokens2.is_empty() {
            return 1.0;
        }

        let set1: AHashSet<_> = tokens1.into_iter().collect();
        let set2: AHashSet<_> = tokens2.into_iter().collect();

        let inter = set1.intersection(&set2).count() as f64;
        let uni = set1.union(&set2).count() as f64;

        if uni == 0.0 {
            0.0
        } else {
            (inter / uni).max(0.0).min(1.0)
        }
    }

    fn jaccard_similarity_chars(&self, s1: &str, s2: &str) -> f64 {
        let set1: AHashSet<char> = s1.chars().collect();
        let set2: AHashSet<char> = s2.chars().collect();

        let intersection = set1.intersection(&set2).count() as f64;
        let union = set1.union(&set2).count() as f64;

        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    /// 字符集 SimHash 相似度
    fn simhash_similarity_chars(&self, s1: &str, s2: &str) -> f64 {
        let tokens1: Vec<String> = s1.chars().map(|c| c.to_string()).collect();
        let tokens2: Vec<String> = s2.chars().map(|c| c.to_string()).collect();

        let h1 = self.simhash_tokens(&tokens1);
        let h2 = self.simhash_tokens(&tokens2);

        let xor = (h1 ^ h2).count_ones() as f64;
        1.0 - (xor / 64.0)
    }

    fn simhash_similarity_tokens(&self, tokens1: &[String], tokens2: &[String]) -> f64 {
        let h1 = self.simhash_tokens(&tokens1);
        let h2 = self.simhash_tokens(&tokens2);

        let xor = (h1 ^ h2).count_ones() as f64;
        1.0 - (xor / 64.0)
    }

    fn simhash_tokens(&self, tokens: &[String]) -> u64 {
        let mut v = [0i32; 64];
        if tokens.is_empty() {
            return 0u64;
        }

        let mut freq: AHashMap<String, i32> = AHashMap::new();
        for t in tokens.iter() {
            *freq.entry(t.clone()).or_insert(0) += 1;
        }

        for (tok, w) in freq.into_iter() {
            let h = self.hash_str(&tok);
            for i in 0..64 {
                let bit = ((h >> i) & 1) as i32;
                v[i] += if bit == 1 { w } else { -w };
            }
        }

        let mut fingerprint = 0u64;
        for i in 0..64 {
            if v[i] > 0 {
                fingerprint |= 1u64 << i;
            }
        }
        fingerprint
    }

    fn hash_str(&self, s: &str) -> u64 {
        let mut hasher = self.hasher_builder.build_hasher();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let ts = TextSimilarity::new("levenshtein").unwrap();

        let s = "4#冷却塔（ACLQ1-1-1）";
        let normalized = ts.normalize(s);
        println!("normalized: {normalized}");
        assert_eq!(normalized, "冷却塔"); // 分词去除括号及编号后按空格分开的 token

        let s2 = "冷却的塔";
        let normalized2 = ts.normalize(s2);
        // "的" 是停用词，应该被去掉
        assert_eq!(normalized2, "冷却 塔");
    }

    #[test]
    fn test_levenshtein_similarity() {
        let ts = TextSimilarity::new("levenshtein").unwrap();
        let s1 = "冷却塔";
        let s2 = "冷却的塔";
        let sim = ts.calculate(s1, s2).unwrap();
        println!("levenshtein sim: {sim}");
        assert!(sim > 0.7); // 相似度较高

        let s3 = "冷却塔1";
        let s4 = "4#冷却塔1（ACLQ1-1-1）";
        let sim1 = ts.calculate(s3, s4).unwrap();
        println!("levenshtein sim1: {sim1}");
        assert!(sim1 > 0.7);
    }

    #[test]
    fn test_jaccard_similarity() {
        let ts = TextSimilarity::new("jaccard").unwrap();
        let s1 = "冷却塔1";
        let s2 = "冷却的塔";
        let sim = ts.calculate(s1, s2).unwrap();
        println!("jaccard sim: {sim}");
        assert!(sim > 0.7); // Jaccard 认为两个集合重合度较高

        let s3 = "冷却塔1";
        let s4 = "4#冷却塔（ACLQ1-1-1）";
        let sim1 = ts.calculate(s3, s4).unwrap();
        println!("jaccard sim1: {sim1}");
        assert!(sim1 >= 0.6);
    }

    #[test]
    fn test_simhash_similarity() {
        let ts = TextSimilarity::new("simhash").unwrap();
        let s1 = "冷却塔";
        let s2 = "4#冷却塔（ACLQ1-1-1）";
        let sim = ts.calculate(s1, s2).unwrap();
        println!("simhash sim: {sim}");
        assert!(sim > 0.8); // SimHash 对相似文本判定较高

        let s3 = "冷却塔1";
        let s4 = "4#冷却塔（ACLQ1-1-1）";
        let sim1 = ts.calculate(s3, s4).unwrap();
        println!("simhash sim1: {sim1}");
        assert!(sim1 > 0.7);
    }

    #[test]
    fn test_batch_calculate() {
        let ts = TextSimilarity::new("levenshtein").unwrap();
        let pairs = vec![
            ("冷却塔".to_string(), "冷却的塔".to_string()),
            ("设备编号1".to_string(), "设备编号2".to_string()),
        ];
        let results = ts.batch_calculate(pairs).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0] > 0.7);
        assert!(results[1] > 0.7);
    }
}
