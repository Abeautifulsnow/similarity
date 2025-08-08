use ahash::AHashSet;
use ahash::RandomState;
use lazy_static::lazy_static;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rapidfuzz::distance::levenshtein;
use std::hash::{BuildHasher, Hash, Hasher};
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

lazy_static! {
    static ref HASHER_BUILDER: RandomState = RandomState::with_seeds(1, 2, 3, 4);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimType {
    Levenshtein,
    Jaccard,
    Simhash,
}

#[pyclass]
pub struct TextSimilaritySimple {
    method: SimType,
}

#[pymethods]
impl TextSimilaritySimple {
    #[new]
    fn new(method: &str) -> PyResult<Self> {
        let method = match method.to_lowercase().as_str() {
            "levenshtein" => SimType::Levenshtein,
            "jaccard" => SimType::Jaccard,
            "simhash" => SimType::Simhash,
            _ => {
                return Err(PyValueError::new_err(
                    "Unsupported similarity method. Use 'levenshtein', 'jaccard' or 'simhash'",
                ));
            }
        };

        Ok(TextSimilaritySimple { method })
    }

    /// 计算文本相似度（同时考虑字符级和 token 级）
    fn calculate(&self, s1: &str, s2: &str) -> PyResult<f64> {
        let n1 = self.normalize(s1);
        let n2 = self.normalize(s2);

        if n1.is_empty() && n2.is_empty() {
            return Ok(1.0);
        }

        let char_now = Instant::now();
        let char_sim = match self.method {
            SimType::Levenshtein => self.levenshtein_similarity(&n1, &n2),
            SimType::Jaccard => self.jaccard_similarity_chars(&n1, &n2),
            SimType::Simhash => self.simhash_similarity_chars(&n1, &n2),
        };
        let char_elapsed = char_now.elapsed();
        println!("Simple calculation time: {char_elapsed:?}");

        Ok(char_sim)
    }

    /// 批量计算相似度
    fn batch_calculate(&self, pairs: Vec<(String, String)>) -> PyResult<Vec<f64>> {
        let mut results = Vec::with_capacity(pairs.len());
        for (a, b) in pairs {
            results.push(self.calculate(&a, &b)?);
        }
        Ok(results)
    }
}

impl TextSimilaritySimple {
    /// 基础文本归一化（仅做 Unicode 标准化和去首尾空格）
    fn normalize(&self, s: &str) -> String {
        // 仅保留 Unicode 标准化（处理中英文混排、全半角等问题）和首尾空格去除
        s.nfkc() // 执行 Unicode 标准化（NFKC 形式）
            .collect::<String>()
            .trim() // 去除首尾空格
            .to_string()
    }

    /// Levenshtein 编辑距离相似度
    fn levenshtein_similarity(&self, s1: &str, s2: &str) -> f64 {
        let dist = levenshtein::distance(s1.chars(), s2.chars());
        let max_len = s1.chars().count().max(s2.chars().count()) as f64;

        if max_len == 0.0 {
            1.0
        } else {
            1.0 - (dist as f64 / max_len)
        }
    }

    /// 字符级 Jaccard 相似度
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

    /// 字符级 SimHash 相似度（与Python版本一致）
    fn simhash_similarity_chars(&self, s1: &str, s2: &str) -> f64 {
        let h1 = self.simhash(s1);
        let h2 = self.simhash(s2);

        // 计算 Hamming 距离并转换为相似度
        let xor = (h1 ^ h2).count_ones() as f64;
        1.0 - (xor / 64.0)
    }

    /// 简易SimHash实现
    fn simhash(&self, text: &str) -> u64 {
        let mut v = [0i32; 64];
        for c in text.chars() {
            let h = self.char_hash(c);
            for i in 0..64 {
                let bit = (h >> i) & 1;
                v[i] += if bit == 1 { 1 } else { -1 };
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

    /// 字符串哈希计算
    fn char_hash(&self, s: char) -> u64 {
        let mut hasher = HASHER_BUILDER.build_hasher();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let ts = TextSimilaritySimple::new("levenshtein").unwrap();

        // 测试 Unicode 标准化（全角转半角等）
        let s = "４＃冷却塔（ＡＣＬＱ１－１－１）　";
        let normalized = ts.normalize(s);
        assert_eq!(normalized, "4#冷却塔（ACLQ1-1-1）"); // 仅标准化和去首尾空格
    }

    #[test]
    fn test_levenshtein_similarity() {
        let ts = TextSimilaritySimple::new("levenshtein").unwrap();

        // 基本相似文本
        let s1 = "冷却塔";
        let s2 = "冷却的塔";
        let sim = ts.calculate(s1, s2).unwrap();
        assert!(sim > 0.6); // 编辑距离为1，长度3 vs 4，相似度 ~0.75

        // 完全相同文本
        let s3 = "测试文本";
        let s4 = "测试文本";
        let sim = ts.calculate(s3, s4).unwrap();
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_jaccard_similarity() {
        let ts = TextSimilaritySimple::new("jaccard").unwrap();

        // 共享大部分字符
        let s1 = "abcde";
        let s2 = "abxde";
        let sim = ts.calculate(s1, s2).unwrap();
        assert_eq!(sim, 4.0 / 6.0); // 交集4，并集6

        // 无共同字符
        let s3 = "abc";
        let s4 = "def";
        let sim = ts.calculate(s3, s4).unwrap();
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_simhash_similarity() {
        let init_now = Instant::now();
        let ts = TextSimilaritySimple::new("simhash").unwrap();
        println!("Simhash init cost: {:?}", init_now.elapsed());

        let now = Instant::now();
        let s1 = "冷却塔1";
        let s2 = "4#冷却塔（ACLQ1-1-1）";
        let sim = ts.calculate(s1, s2).unwrap();
        println!("simhash sim: {sim}");
        assert!(sim > 0.5); // SimHash 对相似文本判定较高
        let elapsed = now.elapsed();
        println!("simhash elapsed: {:?}", elapsed);

        let now1 = Instant::now();
        let s3 = "冷却塔";
        let s4 = "4#冷却塔（ACLQ1-1-1）";
        let sim1 = ts.calculate(s3, s4).unwrap();
        println!("simhash sim1: {sim1}");
        assert!(sim1 > 0.5);
        let elapsed1 = now1.elapsed();
        println!("simhash1 elapsed: {:?}", elapsed1);

        let now2 = Instant::now();
        let s5 = "b-1冷却水泵（APLQ1-1）";
        let s6 = "b-2冷却水泵（ACLQ1-1）";
        let sim2 = ts.calculate(s5, s6).unwrap();
        println!("simhash sim2: {sim1}");
        assert!(sim2 > 0.5);
        let elapsed2 = now2.elapsed();
        println!("simhash2 elapsed: {:?}", elapsed2);
    }

    #[test]
    fn test_batch_calculate() {
        let ts = TextSimilaritySimple::new("levenshtein").unwrap();
        let pairs = vec![
            ("a".to_string(), "a".to_string()),
            ("ab".to_string(), "ac".to_string()),
            ("abc".to_string(), "def".to_string()),
        ];
        let results = ts.batch_calculate(pairs).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], 1.0);
        assert!(results[1] > 0.4);
        assert_eq!(results[2], 0.0);
    }
}
