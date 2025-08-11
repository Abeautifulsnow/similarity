use ahash::{AHashMap, AHashSet, RandomState};
use jieba_rs::Jieba;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rapidfuzz::distance::levenshtein;
use regex::Regex;
use std::hash::{BuildHasher, Hash, Hasher};
use std::string::ToString;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

// 定义哈希种子
const SEED1: u64 = 0xc3ab_a7e8_c40b_5426;
const SEED2: u64 = 0xd2e6_f1a7_b8c9_d0e1;
const SEED3: u64 = 0x1a2b_3c4d_5e6f_7a8b;
const SEED4: u64 = 0x9c8d_7e6f_5a4b_3c2d;

static REMOVE_BRACKETS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[(\[（【][^)\]）】]*[)\]）】]").unwrap());
static NON_WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\p{Script=Han}\p{L}\p{N}]+").unwrap());
static LEADING_NUM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:no\.?|NO\.?\s*)?[#\-]*\d+[#\-]*").unwrap());
static STOPWORDS: LazyLock<AHashSet<String>> = LazyLock::new(|| {
    ["的", "号", "编号"]
        .iter()
        .map(ToString::to_string)
        .collect()
});
static GLOBAL_JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);
static HASHER_BUILDER: LazyLock<RandomState> =
    LazyLock::new(|| RandomState::with_seeds(SEED1, SEED2, SEED3, SEED4));

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
    stopwords: AHashSet<String>,
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
            jieba: GLOBAL_JIEBA.clone(),
            remove_brackets_re: REMOVE_BRACKETS_RE.clone(),
            non_word_re: NON_WORD_RE.clone(),
            leading_num_re: LEADING_NUM_RE.clone(),
            stopwords: STOPWORDS.clone(),
            // 固定种子保证哈希稳定性
            hasher_builder: HASHER_BUILDER.clone(),
        })
    }

    fn select_char(&self, n1: &str, n2: &str) -> f64 {
        match self.method {
            SimType::Levenshtein => self.levenshtein_similarity(n1, n2),
            SimType::Jaccard => self.jaccard_similarity_chars(n1, n2),
            SimType::Simhash => self.simhash_similarity_chars(n1, n2),
        }
    }

    fn calculate(&self, s1: &str, s2: &str) -> f64 {
        // 归一化 - 文本预处理(去除停用词), 生成最终用于比较的标准化文本
        let n1 = self.normalize(s1);
        let n2 = self.normalize(s2);

        if n1.is_empty() && n2.is_empty() {
            return 1.0;
        }

        let char_sim = self.select_char(&n1, &n2);

        // 分词缓存
        // 这里是为了获取分词结果, 用于后续的基于词汇的相似度计算
        let token_vec1 = self.tokenize(&n1);
        let token_vec2 = self.tokenize(&n2);
        let token_sim = match self.method {
            SimType::Levenshtein => char_sim,
            SimType::Jaccard => self.jaccard_similarity_tokens(&token_vec1, &token_vec2),
            SimType::Simhash => self.simhash_similarity_tokens(&token_vec1, &token_vec2),
        };

        token_sim.max(char_sim)
    }

    fn calculate_simple(&self, s1: &str, s2: &str) -> f64 {
        let n1 = self.normalize(s1);
        let n2 = self.normalize(s2);

        if n1.is_empty() && n2.is_empty() {
            return 1.0;
        }

        let char_sim = self.select_char(&n1, &n2);
        char_sim
    }

    fn batch_calculate(&self, pairs: Vec<(String, String)>) -> Vec<f64> {
        let mut results = Vec::with_capacity(pairs.len());
        for (a, b) in pairs {
            results.push(self.calculate(&a, &b));
        }
        results
    }

    fn batch_calculate_simple(&self, pairs: Vec<(String, String)>) -> Vec<f64> {
        let mut results = Vec::with_capacity(pairs.len());
        for (a, b) in pairs {
            results.push(self.calculate_simple(&a, &b));
        }
        results
    }
}

impl TextSimilarity {
    fn normalize(&self, s: &str) -> String {
        // 预分配足够容量，减少动态扩容
        let mut t = String::with_capacity(s.len());
        // 直接在同一个字符串中完成NFKC归一化
        t.extend(s.nfkc());

        // 合并括号和非文字字符替换（用一次replace_all处理多个模式）
        t = self.remove_brackets_re.replace_all(&t, "").into_owned();
        t = self.leading_num_re.replace(&t, "").into_owned();
        // 非文字字符替换为空格（后续可直接split_whitespace）
        t = self.non_word_re.replace_all(&t, " ").into_owned();

        // collapse spaces and trim first, 再做停用词去除会更准
        let collapsed = t.split_whitespace().collect::<Vec<&str>>().join(" ");
        // 利用分词过滤停用词，避免多次replace
        let tokens = self.tokenize(&collapsed);

        tokens.join(" ").trim().to_string()
    }

    fn levenshtein_similarity(&self, s1: &str, s2: &str) -> f64 {
        let dist = levenshtein::distance(s1.chars(), s2.chars());
        let max_len = s1.chars().count().max(s2.chars().count());
        let max_len_f = max_len as f64;
        if max_len_f == 0.0 {
            1.0
        } else {
            1.0 - (dist as f64 / max_len_f)
        }
    }

    fn tokenize(&self, s: &str) -> Vec<String> {
        if s.is_empty() {
            return Vec::<String>::new();
        }

        let chars: Vec<char> = s.chars().collect::<Vec<char>>();
        // 对短文本直接按字符分割（跳过Jieba）
        if chars.len() < 10 {
            chars
                .into_iter()
                .map(|c| c.to_string())
                .filter(|x| !x.is_empty() && !self.stopwords.contains(x))
                .collect()
        } else {
            // 长文本仍用Jieba
            self.jieba
                .cut(s, false)
                .into_iter()
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty() && !self.stopwords.contains(x))
                .collect()
        }
    }

    fn jaccard_similarity_tokens(&self, tokens1: &[String], tokens2: &[String]) -> f64 {
        if tokens1.is_empty() && tokens2.is_empty() {
            return 1.0;
        }

        let set1: AHashSet<_> = tokens1.iter().collect();
        let set2: AHashSet<_> = tokens2.iter().collect();

        let inter = set1.intersection(&set2).count() as f64;
        let uni = set1.union(&set2).count() as f64;

        if uni == 0.0 {
            0.0
        } else {
            (inter / uni).clamp(0.0, 1.0)
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

    /// 字符集 `SimHash` 相似度
    fn simhash_similarity_chars(&self, s1: &str, s2: &str) -> f64 {
        let tokens1: Vec<String> = s1.chars().map(|c| c.to_string()).collect();
        let tokens2: Vec<String> = s2.chars().map(|c| c.to_string()).collect();

        let h1 = self.simhash_tokens(&tokens1);
        let h2 = self.simhash_tokens(&tokens2);

        let xor = f64::from((h1 ^ h2).count_ones());
        1.0 - (xor / 64.0)
    }

    fn simhash_similarity_tokens(&self, tokens1: &[String], tokens2: &[String]) -> f64 {
        let h1 = self.simhash_tokens(tokens1);
        let h2 = self.simhash_tokens(tokens2);

        let xor = f64::from((h1 ^ h2).count_ones());
        1.0 - (xor / 64.0)
    }

    fn simhash_tokens(&self, tokens: &[String]) -> u64 {
        let mut v = [0i32; 64];
        if tokens.is_empty() {
            return 0u64;
        }

        let mut freq: AHashMap<String, i32> = AHashMap::new();
        for t in tokens {
            *freq.entry(t.clone()).or_insert(0) += 1;
        }

        for (tok, w) in freq {
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

#[allow(dead_code)]
/// For compare with `TextSimilarity.simhash_similarity_chars`
fn simhash_fp(s: &str) -> u64 {
    let mut v = [0i32; 64];
    for c in s.chars() {
        let mut hasher = HASHER_BUILDER.build_hasher();
        hasher.write_u32(c as u32);
        let h = hasher.finish();

        for i in 0..64 {
            v[i] += if (h >> i) & 1 == 1 { 1 } else { -1 };
        }
    }

    let mut fingerprint = 0u64;
    for i in 0..64 {
        if v[i] > 0 {
            fingerprint |= 1 << i;
        }
    }
    fingerprint
}

#[allow(dead_code)]
/// For compare with TextSimilarity.simhash_similarity_chars
fn simhash_similarity(s1: &str, s2: &str) -> f64 {
    // NFKC 归一化
    let s1_norm: String = s1.nfkc().collect();
    let s2_norm: String = s2.nfkc().collect();

    let fp1 = simhash_fp(&s1_norm);
    let fp2 = simhash_fp(&s2_norm);

    let xor = (fp1 ^ fp2).count_ones();
    1.0 - f64::from(xor) / 64.0
}

#[cfg(test)]
mod tests {
    // cargo test -r test_simhash_similarity -- --show-output
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_normalize() {
        let ts = TextSimilarity::new("levenshtein").unwrap();

        let s = "4#冷却塔（ACLQ1-1-1）";
        let normalized = ts.normalize(s);
        println!("normalized: {normalized}");
        assert_eq!(normalized, "冷 却 塔"); // 分词去除括号及编号后按空格分开的 token

        let s2 = "冷却的塔";
        let normalized2 = ts.normalize(s2);
        // "的" 是停用词，应该被去掉
        assert_eq!(normalized2, "冷 却 塔");
    }

    #[test]
    fn test_levenshtein_similarity() {
        let ts = TextSimilarity::new("levenshtein").unwrap();
        let s1 = "冷却塔";
        let s2 = "冷却的塔";
        let sim = ts.calculate(s1, s2);
        println!("levenshtein sim: {sim}");
        assert!(sim > 0.7); // 相似度较高

        let s3 = "冷却塔1";
        let s4 = "4#冷却塔1（ACLQ1-1-1）";
        let sim1 = ts.calculate(s3, s4);
        println!("levenshtein sim1: {sim1}");
        assert!(sim1 > 0.7);
    }

    #[test]
    fn test_jaccard_similarity() {
        let ts = TextSimilarity::new("jaccard").unwrap();
        let s1 = "冷却塔1";
        let s2 = "冷却的塔";
        let sim = ts.calculate(s1, s2);
        println!("jaccard sim: {sim}");
        assert!(sim > 0.7); // Jaccard 认为两个集合重合度较高

        let s3 = "冷却塔1";
        let s4 = "4#冷却塔（ACLQ1-1-1）";
        let sim1 = ts.calculate(s3, s4);
        println!("jaccard sim1: {sim1}");
        assert!(sim1 >= 0.6);
    }

    #[test]
    fn test_simhash_similarity() {
        let init_now = Instant::now();
        let ts = TextSimilarity::new("simhash").unwrap();
        println!("Simhash init cost: {:?}", init_now.elapsed());

        let now = Instant::now();
        let s1 = "冷却塔";
        let s2 = "4#冷却塔（ACLQ1-1-1）";
        let sim = ts.calculate(s1, s2);
        println!("simhash sim: {sim}");
        assert!(sim > 0.8); // SimHash 对相似文本判定较高
        let elapsed = now.elapsed();
        println!("simhash elapsed: {elapsed:?}");

        let now1 = Instant::now();
        let s3 = "冷却塔1";
        let s4 = "4#冷却塔（ACLQ1-1-1）";
        let sim1 = ts.calculate(s3, s4);
        println!("simhash sim1: {sim1}");
        assert!(sim1 > 0.7);
        let elapsed1 = now1.elapsed();
        println!("simhash1 elapsed: {elapsed1:?}");
    }

    #[test]
    fn test_simhash_similarity_simple() {
        let init_now = Instant::now();
        let ts = TextSimilarity::new("simhash").unwrap();
        println!("Simhash init cost: {:?}", init_now.elapsed());

        let now = Instant::now();
        let s1 = "冷却塔";
        let s2 = "4#冷却塔（ACLQ1-1-1）";
        let sim = ts.calculate_simple(s1, s2);
        println!("simhash sim: {sim}");
        assert!(sim > 0.8); // SimHash 对相似文本判定较高
        let elapsed = now.elapsed();
        println!("simhash simple elapsed: {elapsed:?}");

        let now1 = Instant::now();
        let s3 = "冷却塔1";
        let s4 = "4#冷却塔（ACLQ1-1-1）";
        let sim1 = ts.calculate_simple(s3, s4);
        println!("simhash sim1: {sim1}");
        assert!(sim1 > 0.7);
        let elapsed1 = now1.elapsed();
        println!("simhash1 simple elapsed: {elapsed1:?}");
    }

    #[test]
    fn test_batch_calculate() {
        let ts = TextSimilarity::new("levenshtein").unwrap();
        let pairs = vec![
            ("冷却塔".to_string(), "冷却的塔".to_string()),
            ("设备编号1".to_string(), "设备编号2".to_string()),
        ];
        let results = ts.batch_calculate(pairs);
        assert_eq!(results.len(), 2);
        assert!(results[0] > 0.7);
        assert!(results[1] > 0.7);
    }

    #[test]
    fn test_string_length() {
        let s2 = "abcde";
        let s3 = "冷却塔1";
        let s4 = "4#冷却塔（ACLQ1-1-1）";
        assert_eq!(s2.len(), s2.chars().count());
        assert_eq!(s3.chars().count(), 4);
        assert_eq!(s4.chars().count(), 16);
    }

    #[test]
    fn test_simhash_fp() {
        let s1 = "冷却塔1";
        let s2 = "4#冷却塔（ACLQ1-1-1）";
        let fp = simhash_similarity(s1, s2);
        println!("simhash fp: {fp}");

        // 测试 simhash 相似度. 该综合效果更好
        let ts = TextSimilarity::new("simhash").unwrap();
        let fp1 = ts.calculate(s1, s2);
        println!("simhash fp1: {fp1}");
    }
}
