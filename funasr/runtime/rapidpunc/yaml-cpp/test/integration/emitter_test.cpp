#include "yaml-cpp/emitterstyle.h"
#include "yaml-cpp/eventhandler.h"
#include "yaml-cpp/yaml.h"  // IWYU pragma: keep
#include "gtest/gtest.h"

namespace YAML {
namespace {

class NullEventHandler : public EventHandler {
  virtual void OnDocumentStart(const Mark&) {}
  virtual void OnDocumentEnd() {}

  virtual void OnNull(const Mark&, anchor_t) {}
  virtual void OnAlias(const Mark&, anchor_t) {}
  virtual void OnScalar(const Mark&, const std::string&, anchor_t,
                        const std::string&) {}

  virtual void OnSequenceStart(const Mark&, const std::string&, anchor_t,
                               EmitterStyle::value /* style */) {}
  virtual void OnSequenceEnd() {}

  virtual void OnMapStart(const Mark&, const std::string&, anchor_t,
                          EmitterStyle::value /* style */) {}
  virtual void OnMapEnd() {}
};

class EmitterTest : public ::testing::Test {
 protected:
  void ExpectEmit(const std::string& expected) {
    EXPECT_EQ(expected, out.c_str());
    EXPECT_TRUE(out.good()) << "Emitter raised: " << out.GetLastError();
    if (expected == out.c_str()) {
      std::stringstream stream(expected);
      Parser parser;
      NullEventHandler handler;
      parser.HandleNextDocument(handler);
    }
  }

  Emitter out;
};

TEST_F(EmitterTest, SimpleScalar) {
  out << "Hello, World!";

  ExpectEmit("Hello, World!");
}

TEST_F(EmitterTest, SimpleQuotedScalar) {
  Node n(Load("\"test\""));
  out << n;
  ExpectEmit("test");
}

TEST_F(EmitterTest, DumpAndSize) {
  Node n(Load("test"));
  EXPECT_EQ("test", Dump(n));
  out << n;
  EXPECT_EQ(4, out.size());
}

TEST_F(EmitterTest, NullScalar) {
  Node n(Load("null"));
  out << n;
  ExpectEmit("~");
}

TEST_F(EmitterTest, AliasScalar) {
  Node n(Load("[&a str, *a]"));
  out << n;
  ExpectEmit("[&1 str, *1]");
}

TEST_F(EmitterTest, StringFormat) {
  out << BeginSeq;
  out.SetStringFormat(SingleQuoted);
  out << "string";
  out.SetStringFormat(DoubleQuoted);
  out << "string";
  out.SetStringFormat(Literal);
  out << "string";
  out << EndSeq;

  ExpectEmit("- 'string'\n- \"string\"\n- |\n  string");
}

TEST_F(EmitterTest, IntBase) {
  out << BeginSeq;
  out.SetIntBase(Dec);
  out << 1024;
  out.SetIntBase(Hex);
  out << 1024;
  out.SetIntBase(Oct);
  out << 1024;
  out << EndSeq;

  ExpectEmit("- 1024\n- 0x400\n- 02000");
}

TEST_F(EmitterTest, NumberPrecision) {
  out.SetFloatPrecision(3);
  out.SetDoublePrecision(2);
  out << BeginSeq;
  out << 3.1425926f;
  out << 53.5893;
  out << 2384626.4338;
  out << EndSeq;

  ExpectEmit("- 3.14\n- 54\n- 2.4e+06");
}

TEST_F(EmitterTest, SimpleSeq) {
  out << BeginSeq;
  out << "eggs";
  out << "bread";
  out << "milk";
  out << EndSeq;

  ExpectEmit("- eggs\n- bread\n- milk");
}

TEST_F(EmitterTest, SimpleFlowSeq) {
  out << Flow;
  out << BeginSeq;
  out << "Larry";
  out << "Curly";
  out << "Moe";
  out << EndSeq;

  ExpectEmit("[Larry, Curly, Moe]");
}

TEST_F(EmitterTest, EmptyFlowSeq) {
  out << Flow;
  out << BeginSeq;
  out << EndSeq;

  ExpectEmit("[]");
}

TEST_F(EmitterTest, EmptyBlockSeqWithBegunContent) {
  out << BeginSeq;
  out << BeginSeq << Comment("comment") << EndSeq;
  out << BeginSeq << Newline << EndSeq;
  out << EndSeq;

  ExpectEmit(R"(-
# comment
  []
-

  [])");
}

TEST_F(EmitterTest, EmptyBlockMapWithBegunContent) {
  out << BeginSeq;
  out << BeginMap << Comment("comment") << EndMap;
  out << BeginMap << Newline << EndMap;
  out << EndSeq;

  ExpectEmit(R"(-  # comment
  {}
-
  {})");
}

TEST_F(EmitterTest, EmptyFlowSeqWithBegunContent) {
  out << Flow;
  out << BeginSeq;
  out << BeginSeq << Comment("comment") << EndSeq;
  out << BeginSeq << Newline << EndSeq;
  out << EndSeq;

  ExpectEmit(R"([[  # comment
  ], [
  ]])");
}

TEST_F(EmitterTest, EmptyFlowMapWithBegunContent) {
  out << Flow;
  out << BeginSeq;
  out << BeginMap << Comment("comment") << EndMap;
  out << BeginMap << Newline << EndMap;
  out << EndSeq;

  ExpectEmit(R"([{  # comment
  }, {
  }])");
}

TEST_F(EmitterTest, NestedBlockSeq) {
  out << BeginSeq;
  out << "item 1";
  out << BeginSeq << "subitem 1"
      << "subitem 2" << EndSeq;
  out << EndSeq;

  ExpectEmit("- item 1\n-\n  - subitem 1\n  - subitem 2");
}

TEST_F(EmitterTest, NestedFlowSeq) {
  out << BeginSeq;
  out << "one";
  out << Flow << BeginSeq << "two"
      << "three" << EndSeq;
  out << EndSeq;

  ExpectEmit("- one\n- [two, three]");
}

TEST_F(EmitterTest, SimpleMap) {
  out << BeginMap;
  out << Key << "name";
  out << Value << "Ryan Braun";
  out << Key << "position";
  out << Value << "3B";
  out << EndMap;

  ExpectEmit("name: Ryan Braun\nposition: 3B");
}

TEST_F(EmitterTest, SimpleFlowMap) {
  out << Flow;
  out << BeginMap;
  out << Key << "shape";
  out << Value << "square";
  out << Key << "color";
  out << Value << "blue";
  out << EndMap;

  ExpectEmit("{shape: square, color: blue}");
}

TEST_F(EmitterTest, MapAndList) {
  out << BeginMap;
  out << Key << "name";
  out << Value << "Barack Obama";
  out << Key << "children";
  out << Value << BeginSeq << "Sasha"
      << "Malia" << EndSeq;
  out << EndMap;

  ExpectEmit("name: Barack Obama\nchildren:\n  - Sasha\n  - Malia");
}

TEST_F(EmitterTest, ListAndMap) {
  out << BeginSeq;
  out << "item 1";
  out << BeginMap;
  out << Key << "pens" << Value << 8;
  out << Key << "pencils" << Value << 14;
  out << EndMap;
  out << "item 2";
  out << EndSeq;

  ExpectEmit("- item 1\n- pens: 8\n  pencils: 14\n- item 2");
}

TEST_F(EmitterTest, NestedBlockMap) {
  out << BeginMap;
  out << Key << "name";
  out << Value << "Fred";
  out << Key << "grades";
  out << Value;
  out << BeginMap;
  out << Key << "algebra" << Value << "A";
  out << Key << "physics" << Value << "C+";
  out << Key << "literature" << Value << "B";
  out << EndMap;
  out << EndMap;

  ExpectEmit(
      "name: Fred\ngrades:\n  algebra: A\n  physics: C+\n  literature: B");
}

TEST_F(EmitterTest, NestedFlowMap) {
  out << Flow;
  out << BeginMap;
  out << Key << "name";
  out << Value << "Fred";
  out << Key << "grades";
  out << Value;
  out << BeginMap;
  out << Key << "algebra" << Value << "A";
  out << Key << "physics" << Value << "C+";
  out << Key << "literature" << Value << "B";
  out << EndMap;
  out << EndMap;

  ExpectEmit("{name: Fred, grades: {algebra: A, physics: C+, literature: B}}");
}

TEST_F(EmitterTest, MapListMix) {
  out << BeginMap;
  out << Key << "name";
  out << Value << "Bob";
  out << Key << "position";
  out << Value;
  out << Flow << BeginSeq << 2 << 4 << EndSeq;
  out << Key << "invincible" << Value << OnOffBool << false;
  out << EndMap;

  ExpectEmit("name: Bob\nposition: [2, 4]\ninvincible: off");
}

TEST_F(EmitterTest, SimpleLongKey) {
  out << LongKey;
  out << BeginMap;
  out << Key << "height";
  out << Value << "5'9\"";
  out << Key << "weight";
  out << Value << 145;
  out << EndMap;

  ExpectEmit("? height\n: 5'9\"\n? weight\n: 145");
}

TEST_F(EmitterTest, SingleLongKey) {
  const std::string shortKey(1024, 'a');
  const std::string longKey(1025, 'a');
  out << BeginMap;
  out << Key << "age";
  out << Value << "24";
  out << LongKey << Key << "height";
  out << Value << "5'9\"";
  out << Key << "weight";
  out << Value << 145;
  out << Key << shortKey;
  out << Value << "1";
  out << Key << longKey;
  out << Value << "1";
  out << EndMap;

  ExpectEmit("age: 24\n? height\n: 5'9\"\nweight: 145\n" + shortKey +
             ": 1\n? " + longKey + "\n: 1");
}

TEST_F(EmitterTest, ComplexLongKey) {
  out << LongKey;
  out << BeginMap;
  out << Key << BeginSeq << 1 << 3 << EndSeq;
  out << Value << "monster";
  out << Key << Flow << BeginSeq << 2 << 0 << EndSeq;
  out << Value << "demon";
  out << EndMap;

  ExpectEmit("? - 1\n  - 3\n: monster\n? [2, 0]\n: demon");
}

TEST_F(EmitterTest, AutoLongKey) {
  out << BeginMap;
  out << Key << BeginSeq << 1 << 3 << EndSeq;
  out << Value << "monster";
  out << Key << Flow << BeginSeq << 2 << 0 << EndSeq;
  out << Value << "demon";
  out << Key << "the origin";
  out << Value << "angel";
  out << EndMap;

  ExpectEmit("? - 1\n  - 3\n: monster\n[2, 0]: demon\nthe origin: angel");
}

TEST_F(EmitterTest, ScalarFormat) {
  out << BeginSeq;
  out << "simple scalar";
  out << SingleQuoted << "explicit single-quoted scalar";
  out << DoubleQuoted << "explicit double-quoted scalar";
  out << "auto-detected\ndouble-quoted scalar";
  out << "a non-\"auto-detected\" double-quoted scalar";
  out << Literal
      << "literal scalar\nthat may span\nmany, many\nlines "
         "and have \"whatever\" crazy\tsymbols that we like";
  out << EndSeq;

  ExpectEmit(
      "- simple scalar\n- 'explicit single-quoted scalar'\n- \"explicit "
      "double-quoted scalar\"\n- \"auto-detected\\ndouble-quoted "
      "scalar\"\n- a "
      "non-\"auto-detected\" double-quoted scalar\n- |\n  literal scalar\n "
      " "
      "that may span\n  many, many\n  lines and have \"whatever\" "
      "crazy\tsymbols that we like");
}

TEST_F(EmitterTest, LiteralWithoutTrailingSpaces) {
  out << YAML::BeginMap;
  out << YAML::Key << "key";
  out << YAML::Value << YAML::Literal;
  out << "expect that with two newlines\n\n"
         "no spaces are emitted in the empty line";
  out << YAML::EndMap;

  ExpectEmit(
      "key: |\n"
      "  expect that with two newlines\n\n"
      "  no spaces are emitted in the empty line");
}

TEST_F(EmitterTest, AutoLongKeyScalar) {
  out << BeginMap;
  out << Key << Literal << "multi-line\nscalar";
  out << Value << "and its value";
  out << EndMap;

  ExpectEmit("? |\n  multi-line\n  scalar\n: and its value");
}

TEST_F(EmitterTest, LongKeyFlowMap) {
  out << Flow;
  out << BeginMap;
  out << Key << "simple key";
  out << Value << "and value";
  out << LongKey << Key << "long key";
  out << Value << "and its value";
  out << EndMap;

  ExpectEmit("{simple key: and value, ? long key: and its value}");
}

TEST_F(EmitterTest, BlockMapAsKey) {
  out << BeginMap;
  out << Key;
  out << BeginMap;
  out << Key << "key" << Value << "value";
  out << Key << "next key" << Value << "next value";
  out << EndMap;
  out << Value;
  out << "total value";
  out << EndMap;

  ExpectEmit("? key: value\n  next key: next value\n: total value");
}

TEST_F(EmitterTest, TaggedBlockMapAsKey) {
  out << BeginMap;
  out << Key;
  out << LocalTag("innerMap");
  out << BeginMap;
  out << Key << "key" << Value << "value";
  out << EndMap;
  out << Value;
  out << "outerValue";
  out << EndMap;

  ExpectEmit(R"(? !innerMap
  key: value
: outerValue)");
}

TEST_F(EmitterTest, TaggedBlockListAsKey) {
  out << BeginMap;
  out << Key;
  out << LocalTag("innerList");
  out << BeginSeq;
  out << "listItem";
  out << EndSeq;
  out << Value;
  out << "outerValue";
  out << EndMap;

  ExpectEmit(R"(? !innerList
  - listItem
: outerValue)");
}

TEST_F(EmitterTest, AliasAndAnchor) {
  out << BeginSeq;
  out << Anchor("fred");
  out << BeginMap;
  out << Key << "name" << Value << "Fred";
  out << Key << "age" << Value << 42;
  out << EndMap;
  out << Alias("fred");
  out << EndSeq;

  ExpectEmit("- &fred\n  name: Fred\n  age: 42\n- *fred");
}

TEST_F(EmitterTest, AliasOnKey) {
  out << BeginSeq;
  out << Anchor("name") << "Name";
  out << BeginMap;
  out << Key << Alias("name") << Value << "Fred";
  out << EndMap;
  out << Flow << BeginMap;
  out << Key << Alias("name") << Value << "Mike";
  out << EndMap;
  out << EndSeq;
  ExpectEmit(R"(- &name Name
- *name : Fred
- {*name : Mike})");
}

TEST_F(EmitterTest, AliasAndAnchorWithNull) {
  out << BeginSeq;
  out << Anchor("fred") << Null;
  out << Alias("fred");
  out << EndSeq;

  ExpectEmit("- &fred ~\n- *fred");
}

TEST_F(EmitterTest, AliasAndAnchorInFlow) {
  out << Flow << BeginSeq;
  out << Anchor("fred");
  out << BeginMap;
  out << Key << "name" << Value << "Fred";
  out << Key << "age" << Value << 42;
  out << EndMap;
  out << Alias("fred");
  out << EndSeq;

  ExpectEmit("[&fred {name: Fred, age: 42}, *fred]");
}

TEST_F(EmitterTest, SimpleVerbatimTag) {
  out << VerbatimTag("!foo") << "bar";

  ExpectEmit("!<!foo> bar");
}

TEST_F(EmitterTest, VerbatimTagInBlockSeq) {
  out << BeginSeq;
  out << VerbatimTag("!foo") << "bar";
  out << "baz";
  out << EndSeq;

  ExpectEmit("- !<!foo> bar\n- baz");
}

TEST_F(EmitterTest, VerbatimTagInFlowSeq) {
  out << Flow << BeginSeq;
  out << VerbatimTag("!foo") << "bar";
  out << "baz";
  out << EndSeq;

  ExpectEmit("[!<!foo> bar, baz]");
}

TEST_F(EmitterTest, VerbatimTagInFlowSeqWithNull) {
  out << Flow << BeginSeq;
  out << VerbatimTag("!foo") << Null;
  out << "baz";
  out << EndSeq;

  ExpectEmit("[!<!foo> ~, baz]");
}

TEST_F(EmitterTest, VerbatimTagInBlockMap) {
  out << BeginMap;
  out << Key << VerbatimTag("!foo") << "bar";
  out << Value << VerbatimTag("!waz") << "baz";
  out << EndMap;

  ExpectEmit("? !<!foo> bar\n: !<!waz> baz");
}

TEST_F(EmitterTest, VerbatimTagInFlowMap) {
  out << Flow << BeginMap;
  out << Key << VerbatimTag("!foo") << "bar";
  out << Value << "baz";
  out << EndMap;

  ExpectEmit("{!<!foo> bar: baz}");
}

TEST_F(EmitterTest, VerbatimTagInFlowMapWithNull) {
  out << Flow << BeginMap;
  out << Key << VerbatimTag("!foo") << Null;
  out << Value << "baz";
  out << EndMap;

  ExpectEmit("{!<!foo> ~: baz}");
}

TEST_F(EmitterTest, VerbatimTagWithEmptySeq) {
  out << VerbatimTag("!foo") << BeginSeq << EndSeq;

  ExpectEmit("!<!foo>\n[]");
}

TEST_F(EmitterTest, VerbatimTagWithEmptyMap) {
  out << VerbatimTag("!bar") << BeginMap << EndMap;

  ExpectEmit("!<!bar>\n{}");
}

TEST_F(EmitterTest, VerbatimTagWithEmptySeqAndMap) {
  out << BeginSeq;
  out << VerbatimTag("!foo") << BeginSeq << EndSeq;
  out << VerbatimTag("!bar") << BeginMap << EndMap;
  out << EndSeq;

  ExpectEmit("- !<!foo>\n  []\n- !<!bar>\n  {}");
}

TEST_F(EmitterTest, ByKindTagWithScalar) {
  out << BeginSeq;
  out << DoubleQuoted << "12";
  out << "12";
  out << TagByKind << "12";
  out << EndSeq;

  ExpectEmit("- \"12\"\n- 12\n- ! 12");
}

TEST_F(EmitterTest, LocalTagInNameHandle) {
  out << LocalTag("a", "foo") << "bar";

  ExpectEmit("!a!foo bar");
}

TEST_F(EmitterTest, LocalTagWithScalar) {
  out << LocalTag("foo") << "bar";

  ExpectEmit("!foo bar");
}

TEST_F(EmitterTest, ComplexDoc) {
  out << BeginMap;
  out << Key << "receipt";
  out << Value << "Oz-Ware Purchase Invoice";
  out << Key << "date";
  out << Value << "2007-08-06";
  out << Key << "customer";
  out << Value;
  out << BeginMap;
  out << Key << "given";
  out << Value << "Dorothy";
  out << Key << "family";
  out << Value << "Gale";
  out << EndMap;
  out << Key << "items";
  out << Value;
  out << BeginSeq;
  out << BeginMap;
  out << Key << "part_no";
  out << Value << "A4786";
  out << Key << "descrip";
  out << Value << "Water Bucket (Filled)";
  out << Key << "price";
  out << Value << 1.47;
  out << Key << "quantity";
  out << Value << 4;
  out << EndMap;
  out << BeginMap;
  out << Key << "part_no";
  out << Value << "E1628";
  out << Key << "descrip";
  out << Value << "High Heeled \"Ruby\" Slippers";
  out << Key << "price";
  out << Value << 100.27;
  out << Key << "quantity";
  out << Value << 1;
  out << EndMap;
  out << EndSeq;
  out << Key << "bill-to";
  out << Value << Anchor("id001");
  out << BeginMap;
  out << Key << "street";
  out << Value << Literal << "123 Tornado Alley\nSuite 16";
  out << Key << "city";
  out << Value << "East Westville";
  out << Key << "state";
  out << Value << "KS";
  out << EndMap;
  out << Key << "ship-to";
  out << Value << Alias("id001");
  out << EndMap;

  ExpectEmit(
      "receipt: Oz-Ware Purchase Invoice\ndate: 2007-08-06\ncustomer:\n  "
      "given: Dorothy\n  family: Gale\nitems:\n  - part_no: A4786\n    "
      "descrip: Water Bucket (Filled)\n    price: 1.47\n    quantity: 4\n  - "
      "part_no: E1628\n    descrip: High Heeled \"Ruby\" Slippers\n    price: "
      "100.27\n    quantity: 1\nbill-to: &id001\n  street: |\n    123 Tornado "
      "Alley\n    Suite 16\n  city: East Westville\n  state: KS\nship-to: "
      "*id001");
}

TEST_F(EmitterTest, STLContainers) {
  out << BeginSeq;
  std::vector<int> primes;
  primes.push_back(2);
  primes.push_back(3);
  primes.push_back(5);
  primes.push_back(7);
  primes.push_back(11);
  primes.push_back(13);
  out << Flow << primes;
  std::map<std::string, int> ages;
  ages["Daniel"] = 26;
  ages["Jesse"] = 24;
  out << ages;
  out << EndSeq;

  ExpectEmit("- [2, 3, 5, 7, 11, 13]\n- Daniel: 26\n  Jesse: 24");
}

TEST_F(EmitterTest, CommentStyle) {
  out.SetPreCommentIndent(1);
  out.SetPostCommentIndent(2);
  out << BeginMap;
  out << Key << "method";
  out << Value << "least squares" << Comment("should we change this method?");
  out << EndMap;

  ExpectEmit("method: least squares #  should we change this method?");
}

TEST_F(EmitterTest, SimpleComment) {
  out << BeginMap;
  out << Key << "method";
  out << Value << "least squares" << Comment("should we change this method?");
  out << EndMap;

  ExpectEmit("method: least squares  # should we change this method?");
}

TEST_F(EmitterTest, MultiLineComment) {
  out << BeginSeq;
  out << "item 1"
      << Comment(
             "really really long\ncomment that couldn't "
             "possibly\nfit on one line");
  out << "item 2";
  out << EndSeq;

  ExpectEmit(
      "- item 1  # really really long\n          # comment that couldn't "
      "possibly\n          # fit on one line\n- item 2");
}

TEST_F(EmitterTest, ComplexComments) {
  out << BeginMap;
  out << LongKey << Key << "long key" << Comment("long key");
  out << Value << "value";
  out << EndMap;

  ExpectEmit("? long key  # long key\n: value");
}

TEST_F(EmitterTest, InitialComment) {
  out << Comment("A comment describing the purpose of the file.");
  out << BeginMap << Key << "key" << Value << "value" << EndMap;

  ExpectEmit("# A comment describing the purpose of the file.\nkey: value");
}

TEST_F(EmitterTest, InitialCommentWithDocIndicator) {
  out << BeginDoc << Comment("A comment describing the purpose of the file.");
  out << BeginMap << Key << "key" << Value << "value" << EndMap;

  ExpectEmit(
      "---\n# A comment describing the purpose of the file.\nkey: value");
}

TEST_F(EmitterTest, CommentInFlowSeq) {
  out << Flow << BeginSeq << "foo" << Comment("foo!") << "bar" << EndSeq;

  ExpectEmit("[foo,  # foo!\nbar]");
}

TEST_F(EmitterTest, CommentInFlowMap) {
  out << Flow << BeginMap;
  out << Key << "foo" << Value << "foo value";
  out << Key << "bar" << Value << "bar value" << Comment("bar!");
  out << Key << "baz" << Value << "baz value" << Comment("baz!");
  out << EndMap;

  ExpectEmit(
      "{foo: foo value, bar: bar value,  # bar!\nbaz: baz value,  # baz!\n}");
}

TEST_F(EmitterTest, Indentation) {
  out << Indent(4);
  out << BeginSeq;
  out << BeginMap;
  out << Key << "key 1" << Value << "value 1";
  out << Key << "key 2" << Value << BeginSeq << "a"
      << "b"
      << "c" << EndSeq;
  out << EndMap;
  out << EndSeq;

  ExpectEmit(
      "-   key 1: value 1\n    key 2:\n        -   a\n        -   b\n        - "
      "  c");
}

TEST_F(EmitterTest, SimpleGlobalSettings) {
  out.SetIndent(4);
  out.SetMapFormat(LongKey);

  out << BeginSeq;
  out << BeginMap;
  out << Key << "key 1" << Value << "value 1";
  out << Key << "key 2" << Value << Flow << BeginSeq << "a"
      << "b"
      << "c" << EndSeq;
  out << EndMap;
  out << EndSeq;

  ExpectEmit("-   ? key 1\n    : value 1\n    ? key 2\n    : [a, b, c]");
}

TEST_F(EmitterTest, GlobalLongKeyOnSeq) {
  out.SetMapFormat(LongKey);

  out << BeginMap;
  out << Key << Anchor("key");
  out << BeginSeq << "a"
      << "b" << EndSeq;
  out << Value << Anchor("value");
  out << BeginSeq << "c"
      << "d" << EndSeq;
  out << Key << Alias("key") << Value << Alias("value");
  out << EndMap;

  ExpectEmit(R"(? &key
  - a
  - b
: &value
  - c
  - d
? *key
: *value)");
}

TEST_F(EmitterTest, GlobalLongKeyOnMap) {
  out.SetMapFormat(LongKey);

  out << BeginMap;
  out << Key << Anchor("key");
  out << BeginMap << "a"
      << "b" << EndMap;
  out << Value << Anchor("value");
  out << BeginMap << "c"
      << "d" << EndMap;
  out << Key << Alias("key") << Value << Alias("value");
  out << EndMap;

  ExpectEmit(R"(? &key
  ? a
  : b
: &value
  ? c
  : d
? *key
: *value)");
}

TEST_F(EmitterTest, GlobalSettingStyleOnSeqNode) {
  Node n(Load(R"(foo:
  - 1
  - 2
  - 3
bar: aa)"));
  out.SetSeqFormat(YAML::Flow);
  out << n;
  ExpectEmit(R"(foo: [1, 2, 3]
bar: aa)");
}

TEST_F(EmitterTest, GlobalSettingStyleOnMapNode) {
  Node n(Load(R"(-
  foo: a
  bar: b
- 2
- 3)"));
  out.SetMapFormat(YAML::Flow);
  out << n;
  ExpectEmit(R"(- {foo: a, bar: b}
- 2
- 3)");
}

TEST_F(EmitterTest, ComplexGlobalSettings) {
  out << BeginSeq;
  out << Block;
  out << BeginMap;
  out << Key << "key 1" << Value << "value 1";
  out << Key << "key 2" << Value;
  out.SetSeqFormat(Flow);
  out << BeginSeq << "a"
      << "b"
      << "c" << EndSeq;
  out << EndMap;
  out << BeginMap;
  out << Key << BeginSeq << 1 << 2 << EndSeq;
  out << Value << BeginMap << Key << "a" << Value << "b" << EndMap;
  out << EndMap;
  out << EndSeq;

  ExpectEmit("- key 1: value 1\n  key 2: [a, b, c]\n- [1, 2]:\n    a: b");
}

TEST_F(EmitterTest, Null) {
  out << BeginSeq;
  out << Null;
  out << BeginMap;
  out << Key << "null value" << Value << Null;
  out << Key << Null << Value << "null key";
  out << EndMap;
  out << EndSeq;

  ExpectEmit("- ~\n- null value: ~\n  ~: null key");
}

TEST_F(EmitterTest, OutputCharset) {
  out << BeginSeq;
  out.SetOutputCharset(EmitNonAscii);
  out << "\x24 \xC2\xA2 \xE2\x82\xAC";
  out.SetOutputCharset(EscapeNonAscii);
  out << "\x24 \xC2\xA2 \xE2\x82\xAC";
  out << EndSeq;

  ExpectEmit("- \x24 \xC2\xA2 \xE2\x82\xAC\n- \"\x24 \\xa2 \\u20ac\"");
}

TEST_F(EmitterTest, EscapedUnicode) {
  out << EscapeNonAscii << "\x24 \xC2\xA2 \xE2\x82\xAC \xF0\xA4\xAD\xA2";

  ExpectEmit("\"$ \\xa2 \\u20ac \\U00024b62\"");
}

TEST_F(EmitterTest, Unicode) {
  out << "\x24 \xC2\xA2 \xE2\x82\xAC \xF0\xA4\xAD\xA2";
  ExpectEmit("\x24 \xC2\xA2 \xE2\x82\xAC \xF0\xA4\xAD\xA2");
}

TEST_F(EmitterTest, DoubleQuotedUnicode) {
  out << DoubleQuoted << "\x24 \xC2\xA2 \xE2\x82\xAC \xF0\xA4\xAD\xA2";
  ExpectEmit("\"\x24 \xC2\xA2 \xE2\x82\xAC \xF0\xA4\xAD\xA2\""); 
}

TEST_F(EmitterTest, EscapedJsonString) {
  out.SetStringFormat(DoubleQuoted);
  out.SetOutputCharset(EscapeAsJson);
  out << "\" \\ "
    "\x01 \x02 \x03 \x04 \x05 \x06 \x07 \x08 \x09 \x0A \x0B \x0C \x0D \x0E \x0F "
    "\x10 \x11 \x12 \x13 \x14 \x15 \x16 \x17 \x18 \x19 \x1A \x1B \x1C \x1D \x1E \x1F "
    "\x24 \xC2\xA2 \xE2\x82\xAC \xF0\xA4\xAD\xA2";

  ExpectEmit(R"("\" \\ \u0001 \u0002 \u0003 \u0004 \u0005 \u0006 \u0007 \b \t )"
             R"(\n \u000b \f \r \u000e \u000f \u0010 \u0011 \u0012 \u0013 )"
             R"(\u0014 \u0015 \u0016 \u0017 \u0018 \u0019 \u001a \u001b )"
             R"(\u001c \u001d \u001e \u001f )"
             "$ \xC2\xA2 \xE2\x82\xAC \xF0\xA4\xAD\xA2\"");
}

TEST_F(EmitterTest, EscapedCharacters) {
  out << BeginSeq 
    << '\x00'
    << '\x0C'
    << '\x0D'
    << EndSeq;

  ExpectEmit("- \"\\x00\"\n- \"\\f\"\n- \"\\r\"");
}

TEST_F(EmitterTest, CharactersEscapedAsJson) {
  out.SetOutputCharset(EscapeAsJson);
  out << BeginSeq
    << '\x00'
    << '\x0C'
    << '\x0D'
    << EndSeq;

  ExpectEmit("- \"\\u0000\"\n- \"\\f\"\n- \"\\r\"");
}

TEST_F(EmitterTest, DoubleQuotedString) {
  out << DoubleQuoted << "\" \\ \n \t \r \b \x15 \xEF\xBB\xBF \x24";
  ExpectEmit("\"\\\" \\\\ \\n \\t \\r \\b \\x15 \\ufeff $\"");
}

struct Foo {
  Foo() : x(0) {}
  Foo(int x_, const std::string& bar_) : x(x_), bar(bar_) {}

  int x;
  std::string bar;
};

Emitter& operator<<(Emitter& out, const Foo& foo) {
  out << BeginMap;
  out << Key << "x" << Value << foo.x;
  out << Key << "bar" << Value << foo.bar;
  out << EndMap;
  return out;
}

TEST_F(EmitterTest, UserType) {
  out << BeginSeq;
  out << Foo(5, "hello");
  out << Foo(3, "goodbye");
  out << EndSeq;

  ExpectEmit("- x: 5\n  bar: hello\n- x: 3\n  bar: goodbye");
}

TEST_F(EmitterTest, UserType2) {
  out << BeginSeq;
  out << Foo(5, "\r");
  out << EndSeq;

  ExpectEmit("- x: 5\n  bar: \"\\r\"");
}

TEST_F(EmitterTest, UserTypeInContainer) {
  std::vector<Foo> fv;
  fv.push_back(Foo(5, "hello"));
  fv.push_back(Foo(3, "goodbye"));
  out << fv;

  ExpectEmit("- x: 5\n  bar: hello\n- x: 3\n  bar: goodbye");
}

template <typename T>
Emitter& operator<<(Emitter& out, const T* v) {
  if (v)
    out << *v;
  else
    out << Null;
  return out;
}

TEST_F(EmitterTest, PointerToInt) {
  int foo = 5;
  int* bar = &foo;
  int* baz = 0;
  out << BeginSeq;
  out << bar << baz;
  out << EndSeq;

  ExpectEmit("- 5\n- ~");
}

TEST_F(EmitterTest, PointerToUserType) {
  Foo foo(5, "hello");
  Foo* bar = &foo;
  Foo* baz = 0;
  out << BeginSeq;
  out << bar << baz;
  out << EndSeq;

  ExpectEmit("- x: 5\n  bar: hello\n- ~");
}

TEST_F(EmitterTest, NewlineAtEnd) {
  out << "Hello" << Newline << Newline;
  ExpectEmit("Hello\n\n");
}

TEST_F(EmitterTest, NewlineInBlockSequence) {
  out << BeginSeq;
  out << "a" << Newline << "b"
      << "c" << Newline << "d";
  out << EndSeq;
  ExpectEmit("- a\n\n- b\n- c\n\n- d");
}

TEST_F(EmitterTest, NewlineInFlowSequence) {
  out << Flow << BeginSeq;
  out << "a" << Newline << "b"
      << "c" << Newline << "d";
  out << EndSeq;
  ExpectEmit("[a,\nb, c,\nd]");
}

TEST_F(EmitterTest, NewlineInBlockMap) {
  out << BeginMap;
  out << Key << "a" << Value << "foo" << Newline;
  out << Key << "b" << Newline << Value << "bar";
  out << LongKey << Key << "c" << Newline << Value << "car";
  out << EndMap;
  ExpectEmit("a: foo\nb:\n  bar\n? c\n\n: car");
}

TEST_F(EmitterTest, NewlineInFlowMap) {
  out << Flow << BeginMap;
  out << Key << "a" << Value << "foo" << Newline;
  out << Key << "b" << Value << "bar";
  out << EndMap;
  ExpectEmit("{a: foo,\nb: bar}");
}

TEST_F(EmitterTest, LotsOfNewlines) {
  out << BeginSeq;
  out << "a" << Newline;
  out << BeginSeq;
  out << "b"
      << "c" << Newline;
  out << EndSeq;
  out << Newline;
  out << BeginMap;
  out << Newline << Key << "d" << Value << Newline << "e";
  out << LongKey << Key << "f" << Newline << Value << "foo";
  out << EndMap;
  out << EndSeq;
  ExpectEmit("- a\n\n-\n  - b\n  - c\n\n\n-\n  d:\n    e\n  ? f\n\n  : foo");
}

TEST_F(EmitterTest, Binary) {
  out << Binary(reinterpret_cast<const unsigned char*>("Hello, World!"), 13);
  ExpectEmit("!!binary \"SGVsbG8sIFdvcmxkIQ==\"");
}

TEST_F(EmitterTest, LongBinary) {
  out << Binary(
      reinterpret_cast<const unsigned char*>(
          "Man is distinguished, not only by his reason, but by this "
          "singular passion from other animals, which is a lust of the "
          "mind, that by a perseverance of delight in the continued and "
          "indefatigable generation of knowledge, exceeds the short "
          "vehemence of any carnal pleasure.\n"),
      270);
  ExpectEmit(
      "!!binary "
      "\"TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieS"
      "B0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIG"
      "x1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbi"
      "B0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZG"
      "dlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS"
      "4K\"");
}

TEST_F(EmitterTest, EmptyBinary) {
  out << Binary(reinterpret_cast<const unsigned char*>(""), 0);
  ExpectEmit("!!binary \"\"");
}

TEST_F(EmitterTest, ColonAtEndOfScalar) {
  out << "a:";
  ExpectEmit("\"a:\"");
}

TEST_F(EmitterTest, ColonAsScalar) {
  out << BeginMap;
  out << Key << "apple" << Value << ":";
  out << Key << "banana" << Value << ":";
  out << EndMap;
  ExpectEmit("apple: \":\"\nbanana: \":\"");
}

TEST_F(EmitterTest, ColonAtEndOfScalarInFlow) {
  out << Flow << BeginMap << Key << "C:" << Value << "C:" << EndMap;
  ExpectEmit("{\"C:\": \"C:\"}");
}

TEST_F(EmitterTest, GlobalBoolFormatting) {
  out << BeginSeq;
  out.SetBoolFormat(UpperCase);
  out.SetBoolFormat(YesNoBool);
  out << true;
  out << false;
  out.SetBoolFormat(TrueFalseBool);
  out << true;
  out << false;
  out.SetBoolFormat(OnOffBool);
  out << true;
  out << false;
  out.SetBoolFormat(LowerCase);
  out.SetBoolFormat(YesNoBool);
  out << true;
  out << false;
  out.SetBoolFormat(TrueFalseBool);
  out << true;
  out << false;
  out.SetBoolFormat(OnOffBool);
  out << true;
  out << false;
  out.SetBoolFormat(CamelCase);
  out.SetBoolFormat(YesNoBool);
  out << true;
  out << false;
  out.SetBoolFormat(TrueFalseBool);
  out << true;
  out << false;
  out.SetBoolFormat(OnOffBool);
  out << true;
  out << false;
  out.SetBoolFormat(ShortBool);
  out.SetBoolFormat(UpperCase);
  out.SetBoolFormat(YesNoBool);
  out << true;
  out << false;
  out.SetBoolFormat(TrueFalseBool);
  out << true;
  out << false;
  out.SetBoolFormat(OnOffBool);
  out << true;
  out << false;
  out << EndSeq;
  ExpectEmit(
      "- YES\n- NO\n- TRUE\n- FALSE\n- ON\n- OFF\n"
      "- yes\n- no\n- true\n- false\n- on\n- off\n"
      "- Yes\n- No\n- True\n- False\n- On\n- Off\n"
      "- Y\n- N\n- Y\n- N\n- Y\n- N");
}

TEST_F(EmitterTest, BoolFormatting) {
  out << BeginSeq;
  out << TrueFalseBool << UpperCase << true;
  out << TrueFalseBool << CamelCase << true;
  out << TrueFalseBool << LowerCase << true;
  out << TrueFalseBool << UpperCase << false;
  out << TrueFalseBool << CamelCase << false;
  out << TrueFalseBool << LowerCase << false;
  out << YesNoBool << UpperCase << true;
  out << YesNoBool << CamelCase << true;
  out << YesNoBool << LowerCase << true;
  out << YesNoBool << UpperCase << false;
  out << YesNoBool << CamelCase << false;
  out << YesNoBool << LowerCase << false;
  out << OnOffBool << UpperCase << true;
  out << OnOffBool << CamelCase << true;
  out << OnOffBool << LowerCase << true;
  out << OnOffBool << UpperCase << false;
  out << OnOffBool << CamelCase << false;
  out << OnOffBool << LowerCase << false;
  out << ShortBool << UpperCase << true;
  out << ShortBool << CamelCase << true;
  out << ShortBool << LowerCase << true;
  out << ShortBool << UpperCase << false;
  out << ShortBool << CamelCase << false;
  out << ShortBool << LowerCase << false;
  out << EndSeq;
  ExpectEmit(
      "- TRUE\n- True\n- true\n- FALSE\n- False\n- false\n"
      "- YES\n- Yes\n- yes\n- NO\n- No\n- no\n"
      "- ON\n- On\n- on\n- OFF\n- Off\n- off\n"
      "- Y\n- Y\n- y\n- N\n- N\n- n");
}

TEST_F(EmitterTest, GlobalNullFormatting) {
  out << Flow << BeginSeq;
  out.SetNullFormat(LowerNull);
  out << Null;
  out.SetNullFormat(UpperNull);
  out << Null;
  out.SetNullFormat(CamelNull);
  out << Null;
  out.SetNullFormat(TildeNull);
  out << Null;
  out << EndSeq;
  ExpectEmit("[null, NULL, Null, ~]");
}

TEST_F(EmitterTest, NullFormatting) {
  out << Flow << BeginSeq;
  out << LowerNull << Null;
  out << UpperNull << Null;
  out << CamelNull << Null;
  out << TildeNull << Null;
  out << EndSeq;
  ExpectEmit("[null, NULL, Null, ~]");
}

TEST_F(EmitterTest, NullFormattingOnNode) {
  Node n(Load("null"));
  out << Flow << BeginSeq;
  out.SetNullFormat(LowerNull);
  out << n;
  out.SetNullFormat(UpperNull);
  out << n;
  out.SetNullFormat(CamelNull);
  out << n;
  out.SetNullFormat(TildeNull);
  out << n;
  out << EndSeq;
  ExpectEmit("[null, NULL, Null, ~]");
}

// TODO: Fix this test.
// TEST_F(EmitterTest, DocStartAndEnd) {
//  out << BeginDoc;
//  out << BeginSeq << 1 << 2 << 3 << EndSeq;
//  out << BeginDoc;
//  out << "Hi there!";
//  out << EndDoc;
//  out << EndDoc;
//  out << EndDoc;
//  out << BeginDoc;
//  out << VerbatimTag("foo") << "bar";
//  ExpectEmit(
//      "---\n- 1\n- 2\n- 3\n---\nHi there!\n...\n...\n...\n---\n!<foo> bar");
//}

TEST_F(EmitterTest, ImplicitDocStart) {
  out << "Hi";
  out << "Bye";
  out << "Oops";
  ExpectEmit("Hi\n---\nBye\n---\nOops");
}

TEST_F(EmitterTest, EmptyString) {
  out << BeginMap;
  out << Key << "key" << Value << "";
  out << EndMap;
  ExpectEmit("key: \"\"");
}

TEST_F(EmitterTest, SingleChar) {
  out << BeginSeq;
  out << 'a';
  out << ':';
  out << (char)0x10;
  out << '\n';
  out << ' ';
  out << '\t';
  out << EndSeq;
  ExpectEmit("- a\n- \":\"\n- \"\\x10\"\n- \"\\n\"\n- \" \"\n- \"\\t\"");
}

TEST_F(EmitterTest, DefaultPrecision) {
  out << BeginSeq;
  out << 1.3125f;
  out << 1.23455810546875;
  out << EndSeq;
  ExpectEmit("- 1.3125\n- 1.23455810546875");
}

TEST_F(EmitterTest, SetPrecision) {
  out << BeginSeq;
  out << FloatPrecision(3) << 1.3125f;
  out << DoublePrecision(6) << 1.23455810546875;
  out << EndSeq;
  ExpectEmit("- 1.31\n- 1.23456");
}

TEST_F(EmitterTest, DashInBlockContext) {
  out << BeginMap;
  out << Key << "key" << Value << "-";
  out << EndMap;
  ExpectEmit("key: \"-\"");
}

TEST_F(EmitterTest, HexAndOct) {
  out << Flow << BeginSeq;
  out << 31;
  out << Hex << 31;
  out << Oct << 31;
  out << EndSeq;
  ExpectEmit("[31, 0x1f, 037]");
}

TEST_F(EmitterTest, CompactMapWithNewline) {
  out << Comment("Characteristics");
  out << BeginSeq;
  out << BeginMap;
  out << Key << "color" << Value << "blue";
  out << Key << "height" << Value << 120;
  out << EndMap;
  out << Newline << Newline;
  out << Comment("Skills");
  out << BeginMap;
  out << Key << "attack" << Value << 23;
  out << Key << "intelligence" << Value << 56;
  out << EndMap;
  out << EndSeq;

  ExpectEmit(
      "# Characteristics\n"
      "- color: blue\n"
      "  height: 120\n"
      "\n"
      "# Skills\n"
      "- attack: 23\n"
      "  intelligence: 56");
}

TEST_F(EmitterTest, ForceSingleQuotedToDouble) {
  out << SingleQuoted << "Hello\nWorld";

  ExpectEmit("\"Hello\\nWorld\"");
}

TEST_F(EmitterTest, QuoteNull) {
  out << "null";

  ExpectEmit("\"null\"");
}

TEST_F(EmitterTest, ValueOfDoubleQuote) {
  out << YAML::BeginMap;
  out << YAML::Key << "foo" << YAML::Value << '"';
  out << YAML::EndMap;

  ExpectEmit("foo: \"\\\"\"");
}

TEST_F(EmitterTest, ValueOfBackslash) {
  out << YAML::BeginMap;
  out << YAML::Key << "foo" << YAML::Value << '\\';
  out << YAML::EndMap;

  ExpectEmit("foo: \"\\\\\"");
}

TEST_F(EmitterTest, Infinity) {
  out << YAML::BeginMap;
  out << YAML::Key << "foo" << YAML::Value
      << std::numeric_limits<float>::infinity();
  out << YAML::Key << "bar" << YAML::Value
      << std::numeric_limits<double>::infinity();
  out << YAML::EndMap;

  ExpectEmit(
	  "foo: .inf\n"
	  "bar: .inf");
}

TEST_F(EmitterTest, NegInfinity) {
  out << YAML::BeginMap;
  out << YAML::Key << "foo" << YAML::Value
      << -std::numeric_limits<float>::infinity();
  out << YAML::Key << "bar" << YAML::Value
      << -std::numeric_limits<double>::infinity();
  out << YAML::EndMap;

  ExpectEmit(
	  "foo: -.inf\n"
	  "bar: -.inf");
}

TEST_F(EmitterTest, NaN) {
  out << YAML::BeginMap;
  out << YAML::Key << "foo" << YAML::Value
      << std::numeric_limits<float>::quiet_NaN();
  out << YAML::Key << "bar" << YAML::Value
      << std::numeric_limits<double>::quiet_NaN();
  out << YAML::EndMap;

  ExpectEmit(
	  "foo: .nan\n"
	  "bar: .nan");
}

TEST_F(EmitterTest, ComplexFlowSeqEmbeddingAMapWithNewLine) { 
  out << YAML::BeginMap;

  out << YAML::Key << "NodeA" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "k" << YAML::Value << YAML::Flow << YAML::BeginSeq;
  out << YAML::BeginMap << YAML::Key << "i" << YAML::Value << 0 << YAML::EndMap
      << YAML::Newline;
  out << YAML::BeginMap << YAML::Key << "i" << YAML::Value << 1 << YAML::EndMap
      << YAML::Newline;
  out << YAML::EndSeq;
  out << YAML::EndMap;

  out << YAML::Key << "NodeB" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "k" << YAML::Value << YAML::Flow << YAML::BeginSeq;
  out << YAML::BeginMap << YAML::Key << "i" << YAML::Value << 0 << YAML::EndMap
      << YAML::Newline;
  out << YAML::BeginMap << YAML::Key << "i" << YAML::Value << 1 << YAML::EndMap
      << YAML::Newline;
  out << YAML::EndSeq;
  out << YAML::EndMap;

  out << YAML::EndMap;

  ExpectEmit(R"(NodeA:
  k: [{i: 0},
    {i: 1},
    ]
NodeB:
  k: [{i: 0},
    {i: 1},
    ])");
}

TEST_F(EmitterTest, ComplexFlowSeqEmbeddingAMapWithNewLineUsingAliases) {
  out << BeginMap;

  out << Key << "Node" << Anchor("Node") << Value << BeginMap;
  out << Key << "k" << Value << Flow << BeginSeq;
  out << BeginMap << Key << "i" << Value << 0 << EndMap;
  out << YAML::Newline;
  out << BeginMap << Key << "i" << Value << 1 << EndMap;
  out << YAML::Newline;
  out << EndSeq << EndMap;

  out << Key << "NodeA" << Alias("Node");
  out << Key << "NodeB" << Alias("Node");

  out << EndMap;

  ExpectEmit(R"(Node: &Node
  k: [{i: 0},
    {i: 1},
    ]
NodeA: *Node
NodeB: *Node)");
}

TEST_F(EmitterTest, ComplexFlowSeqEmbeddingAMapUsingAliases) {
  out << BeginMap;

  out << Key << "Node" << Anchor("Node") << Value << BeginMap;
  out << Key << "k" << Value << Flow << BeginSeq;
  out << BeginMap << Key << "i" << Value << 0 << EndMap;
  out << BeginMap << Key << "i" << Value << 1 << EndMap;
  out << EndSeq << EndMap;

  out << Key << "NodeA" << Alias("Node");
  out << Key << "NodeB" << Alias("Node");

  out << EndMap;

  ExpectEmit(R"(Node: &Node
  k: [{i: 0}, {i: 1}]
NodeA: *Node
NodeB: *Node)");
}

TEST_F(EmitterTest, ComplexFlowSeqEmbeddingAMapWithNewLineUsingAliases2) {
  out << BeginMap;

  out << Key << "Seq" << Anchor("Seq") << Flow << BeginSeq;
  out << BeginMap << Key << "i" << Value << 0 << EndMap;
  out << YAML::Newline;
  out << BeginMap << Key << "i" << Value << 1 << EndMap;
  out << YAML::Newline;
  out << EndSeq;

  out << Key << "NodeA" << Value << BeginMap;
  out << Key << "k" << Value << Alias("Seq") << EndMap;
  out << Key << "NodeB" << Value << BeginMap;
  out << Key << "k" << Value << Alias("Seq") << EndMap;

  out << EndMap;

  ExpectEmit(R"(Seq: &Seq [{i: 0},
  {i: 1},
  ]
NodeA:
  k: *Seq
NodeB:
  k: *Seq)");
}

TEST_F(EmitterTest, ComplexFlowSeqEmbeddingAMapUsingAliases2) {
  out << BeginMap;

  out << Key << "Seq" << Anchor("Seq") << Value << Flow << BeginSeq;
  out << BeginMap << Key << "i" << Value << 0 << EndMap;
  out << BeginMap << Key << "i" << Value << 1 << EndMap;
  out << EndSeq;

  out << Key << "NodeA" << Value << BeginMap;
  out << Key << "k" << Value << Alias("Seq") << EndMap;
  out << Key << "NodeB" << Value << BeginMap;
  out << Key << "k" << Value << Alias("Seq") << EndMap;

  out << EndMap;

  ExpectEmit(R"(Seq: &Seq [{i: 0}, {i: 1}]
NodeA:
  k: *Seq
NodeB:
  k: *Seq)");
}

TEST_F(EmitterTest, ComplexFlowSeqEmbeddingAMapWithNewLineUsingAliases3) {
  out << BeginMap;

  out << Key << "Keys" << Value << Flow << BeginSeq;
  out << Anchor("k0") << BeginMap << Key << "i" << Value << 0 << EndMap
      << Newline;
  out << Anchor("k1") << BeginMap << Key << "i" << Value << 1 << EndMap
      << Newline;
  out << EndSeq;

  out << Key << "NodeA" << Value << BeginMap;
  out << Key << "k" << Value << Flow << BeginSeq;
  out << Alias("k0") << Newline << Alias("k1") << Newline;
  out << EndSeq << EndMap;

  out << Key << "NodeB" << Value << BeginMap;
  out << Key << "k" << Value << Flow << BeginSeq;
  out << Alias("k0") << Newline << Alias("k1") << Newline;
  out << EndSeq << EndMap;

  out << EndMap;

  ExpectEmit(R"(Keys: [&k0 {i: 0},
&k1 {i: 1},
  ]
NodeA:
  k: [*k0,
  *k1,
    ]
NodeB:
  k: [*k0,
  *k1,
    ])");
}

TEST_F(EmitterTest, ComplexFlowSeqEmbeddingAMapUsingAliases3a) {
  out << BeginMap;

  out << Key << "Keys" << Value << BeginSeq;
  out << Anchor("k0") << BeginMap << Key << "i" << Value << 0 << EndMap;
  out << Anchor("k1") << BeginMap << Key << "i" << Value << 1 << EndMap;
  out << EndSeq;

  out << Key << "NodeA" << Value << BeginMap;
  out << Key << "k" << Value << Flow << BeginSeq;
  out << Alias("k0") << Alias("k1");
  out << EndSeq << EndMap;

  out << Key << "NodeB" << Value << BeginMap;
  out << Key << "k" << Value << Flow << BeginSeq;
  out << Alias("k0") << Alias("k1");
  out << EndSeq << EndMap;

  out << EndMap;

  ExpectEmit(R"(Keys:
  - &k0
    i: 0
  - &k1
    i: 1
NodeA:
  k: [*k0, *k1]
NodeB:
  k: [*k0, *k1])");
}

TEST_F(EmitterTest, ComplexFlowSeqEmbeddingAMapUsingAliases3b) {
  out << BeginMap;

  out << Key << "Keys" << Value << Flow << BeginSeq;
  out << Anchor("k0") << BeginMap << Key << "i" << Value << 0 << EndMap;
  out << Anchor("k1") << BeginMap << Key << "i" << Value << 1 << EndMap;
  out << EndSeq;

  out << Key << "NodeA" << Value << BeginMap;
  out << Key << "k" << Value << Flow << BeginSeq;
  out << Alias("k0") << Alias("k1");
  out << EndSeq << EndMap;

  out << Key << "NodeB" << Value << BeginMap;
  out << Key << "k" << Value << Flow << BeginSeq;
  out << Alias("k0") << Alias("k1");
  out << EndSeq << EndMap;

  out << EndMap;

  ExpectEmit(R"(Keys: [&k0 {i: 0}, &k1 {i: 1}]
NodeA:
  k: [*k0, *k1]
NodeB:
  k: [*k0, *k1])");
}

TEST_F(EmitterTest, AnchorEncoding) {
  Node node;
  node["--- &$ [*$]1"] = 1;
  out << node;
  ExpectEmit("\"--- &$ [*$]1\": 1");
  Node reparsed = YAML::Load(out.c_str());
  EXPECT_EQ(reparsed["--- &$ [*$]1"].as<int>(), 1);
}

class EmitterErrorTest : public ::testing::Test {
 protected:
  void ExpectEmitError(const std::string& expectedError) {
    ASSERT_FALSE(out.good()) << "Emitter cleanly produced: " << out.c_str();
    EXPECT_EQ(expectedError, out.GetLastError());
  }

  Emitter out;
};

TEST_F(EmitterErrorTest, BadLocalTag) {
  out << LocalTag("e!far") << "bar";

  ExpectEmitError("invalid tag");
}

TEST_F(EmitterErrorTest, BadTagAndTag) {
  out << VerbatimTag("!far") << VerbatimTag("!foo") << "bar";
  ExpectEmitError(ErrorMsg::INVALID_TAG);
}

TEST_F(EmitterErrorTest, BadAnchorAndAnchor) {
  out << Anchor("far") << Anchor("foo") << "bar";
  ExpectEmitError(ErrorMsg::INVALID_ANCHOR);
}

TEST_F(EmitterErrorTest, BadEmptyAnchorOnGroup) {
  out << BeginSeq << "bar" << Anchor("foo") << EndSeq;
  ExpectEmitError(ErrorMsg::INVALID_ANCHOR);
}

TEST_F(EmitterErrorTest, BadEmptyTagOnGroup) {
  out << BeginSeq << "bar" << VerbatimTag("!foo") << EndSeq;
  ExpectEmitError(ErrorMsg::INVALID_TAG);
}

TEST_F(EmitterErrorTest, ExtraEndSeq) {
  out << BeginSeq;
  out << "Hello";
  out << "World";
  out << EndSeq;
  out << EndSeq;

  ExpectEmitError(ErrorMsg::UNEXPECTED_END_SEQ);
}

TEST_F(EmitterErrorTest, ExtraEndMap) {
  out << BeginMap;
  out << Key << "Hello" << Value << "World";
  out << EndMap;
  out << EndMap;

  ExpectEmitError(ErrorMsg::UNEXPECTED_END_MAP);
}

TEST_F(EmitterErrorTest, InvalidAnchor) {
  out << BeginSeq;
  out << Anchor("new\nline") << "Test";
  out << EndSeq;

  ExpectEmitError(ErrorMsg::INVALID_ANCHOR);
}

TEST_F(EmitterErrorTest, InvalidAlias) {
  out << BeginSeq;
  out << Alias("new\nline");
  out << EndSeq;

  ExpectEmitError(ErrorMsg::INVALID_ALIAS);
}

}  // namespace
}  // namespace YAML
