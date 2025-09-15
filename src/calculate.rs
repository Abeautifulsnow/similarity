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
// 短文本阈值
const SHORT_TEXT_THRESHOLD: usize = 8;

static REMOVE_BRACKETS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[(\[（【][^)\]）】]*[)\]）】]").unwrap());
static NON_WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\p{Script=Han}\p{L}\p{N}]+").unwrap());
static LEADING_NUM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:no\.?|NO\.?\s*)?[#\-]*\d+[#\-]*").unwrap());
static STOPWORDS: LazyLock<AHashSet<String>> = LazyLock::new(|| {
    ["的", "号", "编号", "和", "与", "了", "也", "很", "得", "地"]
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
    Cosine,
    PartialTokenSortRatio, // ✅ 词重排局部匹配（最强）
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
            "cosine" => SimType::Cosine,
            "partial_token_sort_ratio" => SimType::PartialTokenSortRatio, // ✅ 默认最强算法
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid similarity method '{method}'. Valid options are: 'levenshtein', 'jaccard', 'simhash', 'cosine', 'partial_token_sort_ratio'"
                )));
            }
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
        let (long, short) = if n1.chars().count() >= n2.chars().count() {
            (n1, n2)
        } else {
            (n2, n1)
        };

        match self.method {
            SimType::Levenshtein => self.levenshtein_similarity(n1, n2),
            SimType::Jaccard => self.jaccard_similarity_chars(n1, n2),
            SimType::Simhash => self.simhash_similarity_chars(n1, n2),
            SimType::Cosine => self.cosine_similarity_chars(n1, n2),
            SimType::PartialTokenSortRatio => self.partial_token_sort_ratio(long, short),
        }
    }

    fn calculate_similarity(&self, s1: &str, s2: &str, use_tokens: bool) -> f64 {
        // 归一化 - 文本预处理(去除停用词), 生成最终用于比较的标准化文本
        let n1 = self.normalize(s1);
        let n2 = self.normalize(s2);

        if n1.is_empty() && n2.is_empty() {
            return 1.0;
        }

        // // ✅ 对短文本（<20字符）强制用 partial_token_sort_ratio
        // let len1 = n1.chars().count();
        // let len2 = n2.chars().count();
        // if len1 < 20 || len2 < 20 {
        //     return self.partial_token_sort_ratio(
        //         if len1 >= len2 { &n1 } else { &n2 },
        //         if len1 < len2 { &n1 } else { &n2 },
        //     );
        // }

        let char_sim = self.select_char(&n1, &n2);
        if !use_tokens {
            return char_sim;
        }

        // 分词缓存
        // 这里是为了获取分词结果, 用于后续的基于词汇的相似度计算
        let token_vec1 = self.tokenize(&n1);
        let token_vec2 = self.tokenize(&n2);
        let token_sim = match self.method {
            SimType::Levenshtein => char_sim,
            SimType::Jaccard => self.jaccard_similarity_tokens(&token_vec1, &token_vec2),
            SimType::Simhash => self.simhash_similarity_tokens(&token_vec1, &token_vec2),
            SimType::Cosine => self.cosine_similarity_tokens(&token_vec1, &token_vec2),
            SimType::PartialTokenSortRatio => self.partial_token_sort_ratio(&n1, &n2),
        };

        token_sim.max(char_sim)
    }

    fn calculate(&self, s1: &str, s2: &str) -> f64 {
        self.calculate_similarity(s1, s2, true)
    }

    fn calculate_simple(&self, s1: &str, s2: &str) -> f64 {
        self.calculate_similarity(s1, s2, false)
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
    fn lcs_similarity(&self, s1: &str, s2: &str) -> f64 {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let n = chars1.len();
        let m = chars2.len();

        if n == 0 && m == 0 {
            return 1.0;
        }
        if n == 0 || m == 0 {
            return 0.0;
        }

        let mut prev = vec![0; m + 1];
        let mut curr = vec![0; m + 1];

        for i in 1..=n {
            for j in 1..=m {
                if chars1[i - 1] == chars2[j - 1] {
                    curr[j] = prev[j - 1] + 1;
                } else {
                    curr[j] = prev[j].max(curr[j - 1]);
                }
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        let lcs_len = prev[m];
        (2 * lcs_len) as f64 / (n + m) as f64
    }

    fn token_sort_ratio(&self, s1: &str, s2: &str) -> f64 {
        let tokens1 = self.tokenize(s1);
        let tokens2 = self.tokenize(s2);

        if tokens1.is_empty() && tokens2.is_empty() {
            return 1.0;
        }

        // 排序后拼接
        let mut sorted1: Vec<&str> = tokens1.iter().map(|s| s.as_str()).collect();
        let mut sorted2: Vec<&str> = tokens2.iter().map(|s| s.as_str()).collect();
        sorted1.sort_unstable();
        sorted2.sort_unstable();

        // ✅ 拼接时不用空格，避免引入虚假 LCS, 或者用""/\x1F去join
        let str1 = sorted1.join("\x1F");
        let str2 = sorted2.join("\x1F");

        self.lcs_similarity(&str1, &str2)
    }

    fn partial_token_sort_ratio(&self, long: &str, short: &str) -> f64 {
        let short_tokens = self.tokenize(short);
        if short_tokens.is_empty() {
            return 1.0;
        }

        let long_chars: Vec<char> = long.chars().collect();
        let short_len = short.chars().count();
        let long_len = long_chars.len();

        if long_len < short_len {
            return self.token_sort_ratio(long, short);
        }

        let mut max_sim = 0.0;

        // 滑动窗口，窗口长度 = short_len（字符数）
        for i in 0..=(long_len - short_len) {
            let window: String = long_chars[i..i + short_len].iter().collect();
            let sim = self.token_sort_ratio(&window, short);
            if sim > max_sim {
                max_sim = sim;
            }
            if (1.0 - sim) >= f64::EPSILON {
                break;
            }
        }

        max_sim
    }

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
        // let collapsed = t.split_whitespace().collect::<Vec<&str>>().join(" ");
        let collapsed =
            t.split_whitespace()
                .fold(String::with_capacity(t.len()), |mut acc, word| {
                    if !acc.is_empty() {
                        acc.push(' ');
                    }
                    acc.push_str(word);
                    acc
                });
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
        if chars.len() < SHORT_TEXT_THRESHOLD {
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

    /// 词汇集 `Jaccard` 相似度, 入参可能会经过jieba分词, 如果没有经过jieba分词
    /// `那么则与jaccard_similarity_chars效果等价`
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

    /// 词汇集 `Simhash` 相似度, 入参可能会经过jieba分词, 如果没有经过jieba分词
    /// `那么则与simhash_similarity_chars效果等价`
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

        let mut freq: AHashMap<&str, i32> = AHashMap::new();
        for t in tokens {
            *freq.entry(t).or_insert(0) += 1;
        }

        for (tok, w) in freq {
            let h = self.hash_str(tok);
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

    /// 字符集 `Cosine` 相似度
    fn cosine_similarity_chars(&self, s1: &str, s2: &str) -> f64 {
        let tokens1: Vec<String> = s1.chars().map(|c| c.to_string()).collect();
        let tokens2: Vec<String> = s2.chars().map(|c| c.to_string()).collect();

        self.cosine_similarity_tokens(&tokens1, &tokens2)
    }

    /// 词汇集 `Cosine` 相似度
    fn cosine_similarity_tokens(&self, tokens1: &[String], tokens2: &[String]) -> f64 {
        if tokens1.is_empty() && tokens2.is_empty() {
            return 1.0;
        }

        if tokens1.is_empty() || tokens2.is_empty() {
            return 0.0;
        }

        // 计算词频
        let mut freq1: AHashMap<&str, f64> = AHashMap::new();
        let mut freq2: AHashMap<&str, f64> = AHashMap::new();

        for token in tokens1 {
            *freq1.entry(token).or_insert(0.0) += 1.0;
        }

        for token in tokens2 {
            *freq2.entry(token).or_insert(0.0) += 1.0;
        }

        // 计算点积
        let mut dot_product = 0.0;
        for (token, count) in &freq1 {
            dot_product += count * freq2.get(token).unwrap_or(&0.0);
        }

        // 计算向量的模长
        let magnitude1: f64 = freq1.values().map(|x| x * x).sum::<f64>().sqrt();
        let magnitude2: f64 = freq2.values().map(|x| x * x).sum::<f64>().sqrt();

        // 避免除零错误
        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            return 0.0;
        }

        // 计算余弦相似度
        dot_product / (magnitude1 * magnitude2)
    }
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
    fn test_more_case() {
        let ts = TextSimilarity::new("jaccard").unwrap();
        let fp1 = ts.calculate("冷却塔1", "4#冷却塔（ACLQ1-1-1）");
        println!("calculate fp1: {fp1} between: 冷却塔1, 4#冷却塔（ACLQ1-1-1）");

        let fp2 = ts.calculate("风阀", "制冷系统");
        println!("calculate fp2: {fp2} between: 风阀, 制冷系统");

        let fp3 = ts.calculate("风机", "风机");
        println!("calculate fp3: {fp3} between: 风机, 风机");

        let fp4 = ts.calculate("分集水器装置设备", "充电桩装置设备");
        println!("calculate fp4: {fp4} between: 分集水器装置设备, 充电桩装置设备");

        let fp5 = ts.calculate("冷水机组", "冷冻机组");
        println!("calculate fp5: {fp5} between: 冷水机组, 冷冻机组");

        let fp6 = ts.calculate("逆变器", "逆变器_甬金");
        println!("calculate fp6: {fp6} between: 逆变器, 逆变器_甬金");

        let fp7 = ts.calculate("逆变器", "逆变器开关");
        println!("calculate fp7: {fp7} between: 逆变器, 逆变器开关");
    }

    #[test]
    fn test_simhash_similarity_long() {
        let init_now = Instant::now();
        let ts = TextSimilarity::new("simhash").unwrap();
        println!("Simhash init cost: {:?}", init_now.elapsed());

        let s1 = "运行期间无异响";
        let s2 = "加减载设定-冷却泵加减";
        let sim = ts.calculate(s1, s2);
        println!("simhash sim: {sim}");
        assert!(sim > 0.8); // SimHash 对相似文本判定较高

        let s3 = "启动柜无放电声、异味和不均匀机械噪声";
        let s4 = "基载供冷模式下选定冷冻泵";
        let sim1 = ts.calculate(s3, s4);
        println!("simhash sim1: {sim1}");
        assert!(sim1 > 0.7);

        let s5 = "入口压力传感器读数与实际相符";
        let s6 = "入口压力";
        let sim2 = ts.calculate_simple(s5, s6);
        println!("simhash sim2: {sim2}");
    }

    #[test]
    fn test_cosine_similarity_long() {
        let init_now = Instant::now();
        let ts = TextSimilarity::new("cosine").unwrap();
        println!("Cosine init cost: {:?}", init_now.elapsed());

        let s1 = "运行期间无异响";
        let s2 = "加减载设定-冷却泵加减";
        let sim = ts.calculate_simple(s1, s2);
        println!("cosine sim: {sim}");

        let s3 = "启动柜无放电声、异味和不均匀机械噪声";
        let s4 = "基载供冷模式下选定冷冻泵";
        let sim1 = ts.calculate_simple(s3, s4);
        println!("cosine sim1: {sim1}");

        let s5 = "入口压力传感器读数与实际相符";
        let s6 = "入口压力";
        let sim2 = ts.calculate_simple(s5, s6);
        println!("cosine sim2: {sim2}");

        let s7 = "启动柜无放电声、异味和不均匀机械噪声";
        let s8 = "运行时间";
        let sim3 = ts.calculate_simple(s7, s8);
        println!("cosine sim3: {sim3}");
    }

    #[test]
    fn test_levenshtein_similarity_long() {
        let init_now = Instant::now();
        let ts = TextSimilarity::new("levenshtein").unwrap();
        println!("Levenshtein init cost: {:?}", init_now.elapsed());

        let s1 = "运行期间无异响";
        let s2 = "加减载设定-冷却泵加减";
        let sim = ts.calculate_simple(s1, s2);
        println!("Levenshtein sim: {sim}");

        let s3 = "启动柜无放电声、异味和不均匀机械噪声";
        let s4 = "基载供冷模式下选定冷冻泵";
        let sim1 = ts.calculate_simple(s3, s4);
        println!("Levenshtein sim1: {sim1}");

        let s5 = "入口的压力";
        let s6 = "入口压力";
        let sim2 = ts.calculate_simple(s5, s6);
        println!("Levenshtein sim2: {sim2}");

        let s7 = "启动柜无放电声、异味和不均匀机械噪声";
        let s8 = "运行时间";
        let sim3 = ts.calculate_simple(s7, s8);
        println!("Levenshtein sim3: {sim3}");

        let s9 = "算法记录";
        let s10 = "算法负荷记录";
        let sim4 = ts.calculate_simple(s9, s10);
        println!("Levenshtein sim4: {sim4}");

        let s11 = "北京上海";
        let s12 = "上海北京";
        let sim5 = ts.calculate_simple(s11, s12);
        println!("Levenshtein sim5: {sim5}");
    }

    #[test]
    fn test_new_algorithms() {
        let ts = TextSimilarity::new("partial_token_sort_ratio").unwrap();

        let cases = vec![
            ("分析出口压力的最大值和最小值", "出口压力", 1.0),
            ("入口的压力传感器读数与实际相符", "入口压力", 1.0),
            ("出口压力", "出口的压力", 0.85),
            ("电流值和频率符合设备额定要求", "频率", 1.0),
            ("启泵延时T11RN_VAI", "启泵延时T11R_VAI", 0.9),
            ("负荷修正系数b", "负荷修正系数a", 0.9),
            ("运行期间无异响", "加减载设定-冷却泵加减", 0.0),
            ("分析出口压力的最大值和最小值", "出口压力", 0.9),
            ("出口压力", "出口的压力", 0.85),
            ("电流值及频率符合设备额定要求", "频率", 1.0),
            ("电流值及频率符合设备额定要求", "电流值", 1.0),
            ("入口压力传感器读数与实际相符", "入口压力", 1.0),
            ("入口的压力传感器读数与实际相符", "入口压力", 1.0),
            ("出口压力", "出口压力值", 0.85),
            ("出口的压力", "出口压力", 0.85),
            ("正在执行的工况是否与实际相符", "正在执行工况", 1.0),
            ("算法记录", "算法负荷记录", 1.0),
            ("冷冻泵", "YB-1乙二醇泵（APLD1-1）", 0.0),
            ("b-1冷却水泵（APLQ1-1）", "b-2冷却水泵（ACLQ1-1）", 0.85),
            ("分集水器", "控制单元", 0.0),
            ("逆变器", "逆变器开关", 0.85),
            ("负荷修正系数b", "负荷修正系数a", 0.85),
            ("启泵延时T11RN_VAI", "启泵延时T11R_VAI", 0.9),
            ("出口压力", "入口压力", 0.85),
            ("运行期间无异响", "加减载设定-冷却泵加减", 0.0),
            ("启动柜无放电声、异味和不均匀机械噪声", "用电量", 0.0),
            (
                "启动柜无放电声、异味和不均匀机械噪声",
                "基载供冷模式下选定冷冻泵",
                0.0,
            ),
            ("逆变器", "逆变器_甬金", 0.85),
            ("逆变器", "逆变器开关", 0.85),
            ("上海北京", "北京上海", 0.85),
        ];

        for (s1, s2, _expected_min) in cases {
            let sim = ts.calculate(s1, s2);
            println!("'{s1}' vs '{s2}' = {sim:.3}");
            // assert!(sim >= expected_min, "Failed: '{s1}' vs '{s2}' = {sim:.3}");
        }
    }
}
