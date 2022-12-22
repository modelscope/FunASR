# Chinese Text Normalization

## 1. How To Use
```
python normalize.py --language "zh" --text "text to be normalized"
```

## 2. TN Pipeline
There are 3 components in TN pipeline:
* pre-processing (before tagger)
* non-standard word normalization
* post-processing (after verbalizer)

### 2.1 Pre-Processing
#### Char Width Conversion (全角 -> 半角)
```
苹果ＣＥＯ宣布发布新ＩＰＨＯＮＥ -> 苹果CEO宣布发布新IPHONE
他说：“我们已经吃过了！”。 -> 他说:"我们已经吃过了!".
```
* covers English letters, digits, punctuations and some symbols
* the complete mapping table `data/char/fullwidth_to_halfwidth.tsv`

#### Denylist (Removal)
Sometime you may want to remove certain things like interjections/fillers "啊", "呃" etc
```
呃这个呃啊我不知道 -> 这个我不知道
```
* customizable via `data/denylist/denylist.tsv`


### 2.2 Non-Standard-Words(NSW) normalization
#### Numbers
```
共465篇，约315万字 -> 共四百六十五篇，约三百一十五万字
共计6.42万人 -> 共计六点四二万人
同比升高0.6个百分点 -> 同比升高零点六个百分点
```

#### Fraction
```
总量的1/5以上 -> 总量的五分之一以上
相当于头发丝的1/16 -> 相当于头发丝的十六分之一
3/2是一个假分数 -> 二分之三是一个假分数
```

#### Percentage
```
同比增长6.3% -> 同比增长百分之六点三
增幅0.4% -> 增幅百分之零点四
```

#### Date
```
2002/01/28 -> 二零零二年一月二十八日
2002-01-28 -> 二零零二年一月二十八日
2002.01.28 -> 二零零二年一月二十八日
2002/01 -> 二零零二年一月
```

#### Time
```
8月16号12:00之前 -> 八月十六号十二点之前
我是5:02开始的 -> 我是五点零二分开始的
于5:35:36发射 -> 于五点三十五分三十六秒发射
8:00am准时开会 -> 上午八点准时开会
```

#### Math
```
比分定格在78:96 -> 比分定格在七十八比九十六
计算-2的绝对值是2 -> 计算负二的绝对值是二
±2的平方都是4 -> 正负二的平方都是四
```

#### Money
```
价格是￥13.5 -> 价格是十三点五元
价格是$13.5 -> 价格是十三点五美元
价格是A$13.5 -> 价格是十三点五澳元
价格是HKD13.5 -> 价格是十三点五港元
```

#### Measure
```
重达25kg -> 二十五千克
最高气温38°C -> 最高气温三十八摄氏度
实际面积120m² -> 实际面积一百二十平方米
渲染速度10ms一帧 -> 渲染速度十毫秒一帧
```

#### Number series (phone, mobile numbers)
```
可以打我手机13501234567 -> 可以打我手机一三五零一二三四五六七
可以拨打12306来咨询 -> 可以拨打一二三零六来咨询
```

#### Erhua(儿化音) Removal
```
这儿有只鸟儿 -> 这有只鸟
这事儿好办 -> 这事好办
我儿子喜欢这地儿 -> 我儿子喜欢这地
```
* erhua whitelist is customizable via `data/erhua/whitelist.tsv`

#### Whitelist(Replacement)
a set of user-defined hard mapping, i.e. exact-string matching & replacement
```
C E O -> CEO
G P U -> GPU
O2O -> O to O
B2B -> B to B
```
* customizable via `data/whitelist/default.tsv`

### 2.3 Post-Processing
#### Punctuation Removal
If enabled, punctuations are removed.

#### Uppercase or Lowercase Conversion
If enabled, English letters are converted to uppercases / lowercases

#### Out-Of-Vocabulary(OOV) Tagger
If enabled, OOV chars are tagged with `<oov>` and `</oov>`, e.g.:
```
我们안녕 -> 我们<oov>안</oov><oov>녕</oov>
雪の花 -> 雪<oov>の</oov>花
```
* default charset (national standard) [通用规范汉字表](https://zh.wikipedia.org/wiki/通用规范汉字表)
* you can extend charset via `data/char/charset_extension.tsv`

## 3. Credits
Author: Zhenxiang MA @ Tsinghua University

Advisors: [SpeechColab](https://github.com/SpeechColab) organization

The authors of this work would like to thank:
* The authors of foundational libraries like OpenFst & Pynini
* NeMo team and NeMo open-source community
