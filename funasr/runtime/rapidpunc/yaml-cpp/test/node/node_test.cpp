#include "yaml-cpp/node/node.h"
#include "yaml-cpp/emitter.h"
#include "yaml-cpp/node/convert.h"
#include "yaml-cpp/node/detail/impl.h"
#include "yaml-cpp/node/emit.h"
#include "yaml-cpp/node/impl.h"
#include "yaml-cpp/node/iterator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <sstream>

namespace {

// malloc/free based allocator just for testing custom allocators on stl containers
template <class T>
class CustomAllocator : public std::allocator<T> {
  public:
    typedef std::size_t     size_type;
    typedef std::ptrdiff_t  difference_type;
    typedef T*              pointer;
    typedef const T*        const_pointer;
    typedef T&              reference;
    typedef const T&        const_reference;
    typedef T               value_type;
    template<class U> struct rebind { typedef CustomAllocator<U> other; };
    CustomAllocator() : std::allocator<T>() {}
    CustomAllocator(const CustomAllocator& other) : std::allocator<T>(other) {}
    template<class U> CustomAllocator(const CustomAllocator<U>& other) : std::allocator<T>(other) {}
    ~CustomAllocator() {}
    size_type max_size() const { return (std::numeric_limits<std::ptrdiff_t>::max)()/sizeof(T); }
    pointer allocate(size_type num, const void* /*hint*/ = 0) {
      if (num > std::size_t(-1) / sizeof(T)) throw std::bad_alloc();
      return static_cast<pointer>(malloc(num * sizeof(T)));
    }
    void deallocate(pointer p, size_type /*num*/) { free(p); }
};

template <class T> using CustomVector = std::vector<T,CustomAllocator<T>>;
template <class T> using CustomList = std::list<T,CustomAllocator<T>>;
template <class K, class V, class C=std::less<K>> using CustomMap = std::map<K,V,C,CustomAllocator<std::pair<const K,V>>>;
template <class K, class V, class H=std::hash<K>, class P=std::equal_to<K>> using CustomUnorderedMap = std::unordered_map<K,V,H,P,CustomAllocator<std::pair<const K,V>>>;

}  // anonymous namespace

using ::testing::AnyOf;
using ::testing::Eq;

#define EXPECT_THROW_REPRESENTATION_EXCEPTION(statement, message) \
  ASSERT_THROW(statement, RepresentationException);               \
  try {                                                           \
    statement;                                                    \
  } catch (const RepresentationException& e) {                    \
    EXPECT_EQ(e.msg, message);                                    \
  }

namespace YAML {
namespace {
TEST(NodeTest, SimpleScalar) {
  Node node = Node("Hello, World!");
  EXPECT_TRUE(node.IsScalar());
  EXPECT_EQ("Hello, World!", node.as<std::string>());
}

TEST(NodeTest, IntScalar) {
  Node node = Node(15);
  EXPECT_TRUE(node.IsScalar());
  EXPECT_EQ(15, node.as<int>());
}

TEST(NodeTest, SimpleAppendSequence) {
  Node node;
  node.push_back(10);
  node.push_back("foo");
  node.push_back("monkey");
  EXPECT_TRUE(node.IsSequence());
  EXPECT_EQ(3, node.size());
  EXPECT_EQ(10, node[0].as<int>());
  EXPECT_EQ("foo", node[1].as<std::string>());
  EXPECT_EQ("monkey", node[2].as<std::string>());
  EXPECT_TRUE(node.IsSequence());
}

TEST(NodeTest, SequenceElementRemoval) {
  Node node;
  node[0] = "a";
  node[1] = "b";
  node[2] = "c";
  node.remove(1);
  EXPECT_TRUE(node.IsSequence());
  EXPECT_EQ(2, node.size());
  EXPECT_EQ("a", node[0].as<std::string>());
  EXPECT_EQ("c", node[1].as<std::string>());
}

TEST(NodeTest, SequenceElementRemovalSizeCheck) {
  Node node;
  node[0] = "a";
  node[1] = "b";
  node[2] = "c";
  EXPECT_EQ(3, node.size());
  node.remove(1);
  EXPECT_TRUE(node.IsSequence());
  EXPECT_EQ(2, node.size());
  EXPECT_EQ("a", node[0].as<std::string>());
  EXPECT_EQ("c", node[1].as<std::string>());
}

TEST(NodeTest, SequenceFirstElementRemoval) {
  Node node;
  node[0] = "a";
  node[1] = "b";
  node[2] = "c";
  node.remove(0);
  EXPECT_TRUE(node.IsSequence());
  EXPECT_EQ(2, node.size());
  EXPECT_EQ("b", node[0].as<std::string>());
  EXPECT_EQ("c", node[1].as<std::string>());
}

TEST(NodeTest, SequenceFirstElementRemovalSizeCheck) {
  Node node;
  node[0] = "a";
  node[1] = "b";
  node[2] = "c";
  EXPECT_EQ(3, node.size());
  node.remove(0);
  EXPECT_TRUE(node.IsSequence());
  EXPECT_EQ(2, node.size());
  EXPECT_EQ("b", node[0].as<std::string>());
  EXPECT_EQ("c", node[1].as<std::string>());
}

TEST(NodeTest, SequenceLastElementRemoval) {
  Node node;
  node[0] = "a";
  node[1] = "b";
  node[2] = "c";
  node.remove(2);
  EXPECT_TRUE(node.IsSequence());
  EXPECT_EQ(2, node.size());
  EXPECT_EQ("a", node[0].as<std::string>());
  EXPECT_EQ("b", node[1].as<std::string>());
}

TEST(NodeTest, SequenceLastElementRemovalSizeCheck) {
  Node node;
  node[0] = "a";
  node[1] = "b";
  node[2] = "c";
  EXPECT_EQ(3, node.size());
  node.remove(2);
  EXPECT_TRUE(node.IsSequence());
  EXPECT_EQ(2, node.size());
  EXPECT_EQ("a", node[0].as<std::string>());
  EXPECT_EQ("b", node[1].as<std::string>());
}

TEST(NodeTest, NodeAssignment) {
  Node node1;
  Node node2;
  node1[1] = 1;
  node1[2] = 2;
  node1[3] = 3;
  node2 = node1;
  EXPECT_EQ(node1, node2);
  EXPECT_EQ(node1[1], node2[1]);
  EXPECT_EQ(node1[2], node2[2]);
  EXPECT_EQ(node1[3], node2[3]);
}

TEST(NodeTest, EqualRepresentationAfterMoveAssignment) {
  Node node1;
  Node node2;
  std::ostringstream ss1, ss2;
  node1["foo"] = "bar";
  ss1 << node1;
  node2["hello"] = "world";
  node2 = std::move(node1);
  ss2 << node2;
  EXPECT_FALSE(node2["hello"]);
  EXPECT_EQ("bar", node2["foo"].as<std::string>());
  EXPECT_EQ(ss1.str(), ss2.str());
}

TEST(NodeTest, MapElementRemoval) {
  Node node;
  node["foo"] = "bar";
  node.remove("foo");
  EXPECT_TRUE(!node["foo"]);
}

TEST(NodeTest, MapIntegerElementRemoval) {
  Node node;
  node[1] = "hello";
  node[2] = 'c';
  node["foo"] = "bar";
  EXPECT_TRUE(node.IsMap());
  node.remove(1);
  EXPECT_TRUE(node.IsMap());
}

TEST(NodeTest, SimpleAssignSequence) {
  Node node;
  node[0] = 10;
  node[1] = "foo";
  node[2] = "monkey";
  EXPECT_TRUE(node.IsSequence());
  EXPECT_EQ(3, node.size());
  EXPECT_EQ(10, node[0].as<int>());
  EXPECT_EQ("foo", node[1].as<std::string>());
  EXPECT_EQ("monkey", node[2].as<std::string>());
  EXPECT_TRUE(node.IsSequence());
}

TEST(NodeTest, SimpleMap) {
  Node node;
  node["key"] = "value";
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ("value", node["key"].as<std::string>());
  EXPECT_EQ(1, node.size());
}

TEST(NodeTest, MapWithUndefinedValues) {
  Node node;
  node["key"] = "value";
  node["undefined"];
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ("value", node["key"].as<std::string>());
  EXPECT_EQ(1, node.size());

  node["undefined"] = "monkey";
  EXPECT_EQ("monkey", node["undefined"].as<std::string>());
  EXPECT_EQ(2, node.size());
}

TEST(NodeTest, SeqIntoMap) {
  Node node;
  node[0] = "test";
  node[1];
  node[2] = "value";
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ("test", node[0].as<std::string>());
  EXPECT_EQ("value", node[2].as<std::string>());
  EXPECT_EQ(2, node.size());
}

TEST(NodeTest, RemoveUnassignedNode) {
  Node node(NodeType::Map);
  node["key"];
  node.remove("key");
  EXPECT_EQ(0, node.size());
}

TEST(NodeTest, RemoveUnassignedNodeFromMap) {
  Node node(NodeType::Map);
  Node n;
  node[n];
  node.remove(n);
  EXPECT_EQ(0, node.size());
}

TEST(NodeTest, MapForceInsert) {
  Node node;
  Node k1("k1");
  Node k2("k2");
  Node v1("v1");
  Node v2("v2");
  node[k1] = v1;
  node[k2] = v1;
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ("v1", node["k1"].as<std::string>());
  EXPECT_EQ("v1", node["k2"].as<std::string>());
  EXPECT_EQ(2, node.size());

  node.force_insert(k2, v2);
  EXPECT_EQ("v1", node["k1"].as<std::string>());
  EXPECT_EQ("v1", node["k2"].as<std::string>());
  EXPECT_EQ(3, node.size());
}

TEST(NodeTest, UndefinedConstNodeWithFallback) {
  Node node;
  const Node& cn = node;
  EXPECT_EQ(cn["undefined"].as<int>(3), 3);
}

TEST(NodeTest, MapIteratorWithUndefinedValues) {
  Node node;
  node["key"] = "value";
  node["undefined"];

  std::size_t count = 0;
  for (const_iterator it = node.begin(); it != node.end(); ++it)
    count++;
  EXPECT_EQ(1, count);
}

TEST(NodeTest, ConstIteratorOnConstUndefinedNode) {
  Node node;
  const Node& cn = node;
  const Node& undefinedCn = cn["undefined"];

  std::size_t count = 0;
  for (const_iterator it = undefinedCn.begin(); it != undefinedCn.end(); ++it) {
    count++;
  }
  EXPECT_EQ(0, count);
}

TEST(NodeTest, IteratorOnConstUndefinedNode) {
  Node node;
  const Node& cn = node;
  const Node& undefinedCn = cn["undefined"];

  Node& nonConstUndefinedNode = const_cast<Node&>(undefinedCn);

  std::size_t count = 0;
  for (iterator it = nonConstUndefinedNode.begin();
       it != nonConstUndefinedNode.end(); ++it) {
    count++;
  }
  EXPECT_EQ(0, count);
}
  
TEST(NodeTest, InteratorOnSequence) {
  Node node;
  node[0] = "a";
  node[1] = "b";
  node[2] = "c";
  EXPECT_TRUE(node.IsSequence());
  
  std::size_t count = 0;
  for (iterator it = node.begin(); it != node.end(); ++it)
  {
    EXPECT_FALSE(it->IsNull());
    count++;
  }
  EXPECT_EQ(3, count);
}
  
TEST(NodeTest, ConstInteratorOnSequence) {
  Node node;
  node[0] = "a";
  node[1] = "b";
  node[2] = "c";
  EXPECT_TRUE(node.IsSequence());
  
  std::size_t count = 0;
  for (const_iterator it = node.begin(); it != node.end(); ++it)
  {
    EXPECT_FALSE(it->IsNull());
    count++;
  }
  EXPECT_EQ(3, count);
}

TEST(NodeTest, SimpleSubkeys) {
  Node node;
  node["device"]["udid"] = "12345";
  node["device"]["name"] = "iPhone";
  node["device"]["os"] = "4.0";
  node["username"] = "monkey";
  EXPECT_EQ("12345", node["device"]["udid"].as<std::string>());
  EXPECT_EQ("iPhone", node["device"]["name"].as<std::string>());
  EXPECT_EQ("4.0", node["device"]["os"].as<std::string>());
  EXPECT_EQ("monkey", node["username"].as<std::string>());
}

TEST(NodeTest, StdArray) {
  std::array<int, 5> evens{{2, 4, 6, 8, 10}};
  Node node;
  node["evens"] = evens;
  std::array<int, 5> actualEvens = node["evens"].as<std::array<int, 5>>();
  EXPECT_EQ(evens, actualEvens);
}

TEST(NodeTest, StdArrayWrongSize) {
  std::array<int, 3> evens{{2, 4, 6}};
  Node node;
  node["evens"] = evens;
  EXPECT_THROW_REPRESENTATION_EXCEPTION(
      (node["evens"].as<std::array<int, 5>>()), ErrorMsg::BAD_CONVERSION);
}

TEST(NodeTest, StdValrray) {
  std::valarray<int> evens{{2, 4, 6, 8, 10}};
  Node node;
  node["evens"] = evens;
  std::valarray<int> actualEvens = node["evens"].as<std::valarray<int>>();
  for (int i = 0; i < evens.size(); ++i) {
    EXPECT_EQ(evens[i], actualEvens[i]);
  }
}

TEST(NodeTest, StdVector) {
  std::vector<int> primes;
  primes.push_back(2);
  primes.push_back(3);
  primes.push_back(5);
  primes.push_back(7);
  primes.push_back(11);
  primes.push_back(13);

  Node node;
  node["primes"] = primes;
  EXPECT_EQ(primes, node["primes"].as<std::vector<int>>());
}

TEST(NodeTest, StdVectorWithCustomAllocator) {
  CustomVector<int> primes;
  primes.push_back(2);
  primes.push_back(3);
  primes.push_back(5);
  primes.push_back(7);
  primes.push_back(11);
  primes.push_back(13);

  Node node;
  node["primes"] = primes;
  EXPECT_EQ(primes, node["primes"].as<CustomVector<int>>());
}

TEST(NodeTest, StdList) {
  std::list<int> primes;
  primes.push_back(2);
  primes.push_back(3);
  primes.push_back(5);
  primes.push_back(7);
  primes.push_back(11);
  primes.push_back(13);

  Node node;
  node["primes"] = primes;
  EXPECT_EQ(primes, node["primes"].as<std::list<int>>());
}

TEST(NodeTest, StdListWithCustomAllocator) {
  CustomList<int> primes;
  primes.push_back(2);
  primes.push_back(3);
  primes.push_back(5);
  primes.push_back(7);
  primes.push_back(11);
  primes.push_back(13);

  Node node;
  node["primes"] = primes;
  EXPECT_EQ(primes, node["primes"].as<CustomList<int>>());
}

TEST(NodeTest, StdMap) {
  std::map<int, int> squares;
  squares[0] = 0;
  squares[1] = 1;
  squares[2] = 4;
  squares[3] = 9;
  squares[4] = 16;

  Node node;
  node["squares"] = squares;
  std::map<int, int> actualSquares = node["squares"].as<std::map<int, int>>();
  EXPECT_EQ(squares, actualSquares);
}

TEST(NodeTest, StdMapWithCustomAllocator) {
  CustomMap<int,int> squares;
  squares[0] = 0;
  squares[1] = 1;
  squares[2] = 4;
  squares[3] = 9;
  squares[4] = 16;

  Node node;
  node["squares"] = squares;
  CustomMap<int,int> actualSquares = node["squares"].as<CustomMap<int,int>>();
  EXPECT_EQ(squares, actualSquares);
}

TEST(NodeTest, StdUnorderedMap) {
  std::unordered_map<int, int> squares;
  squares[0] = 0;
  squares[1] = 1;
  squares[2] = 4;
  squares[3] = 9;
  squares[4] = 16;

  Node node;
  node["squares"] = squares;
  std::unordered_map<int, int> actualSquares = node["squares"].as<std::unordered_map<int, int>>();
  EXPECT_EQ(squares, actualSquares);
}

TEST(NodeTest, StdUnorderedMapWithCustomAllocator) {
  CustomUnorderedMap<int,int> squares;
  squares[0] = 0;
  squares[1] = 1;
  squares[2] = 4;
  squares[3] = 9;
  squares[4] = 16;

  Node node;
  node["squares"] = squares;
  CustomUnorderedMap<int,int> actualSquares = node["squares"].as<CustomUnorderedMap<int,int>>();
  EXPECT_EQ(squares, actualSquares);
}

TEST(NodeTest, StdPair) {
  std::pair<int, std::string> p;
  p.first = 5;
  p.second = "five";

  Node node;
  node["pair"] = p;
  std::pair<int, std::string> actualP =
      node["pair"].as<std::pair<int, std::string>>();
  EXPECT_EQ(p, actualP);
}

TEST(NodeTest, SimpleAlias) {
  Node node;
  node["foo"] = "value";
  node["bar"] = node["foo"];
  EXPECT_EQ("value", node["foo"].as<std::string>());
  EXPECT_EQ("value", node["bar"].as<std::string>());
  EXPECT_EQ(node["bar"], node["foo"]);
  EXPECT_EQ(2, node.size());
}

TEST(NodeTest, AliasAsKey) {
  Node node;
  node["foo"] = "value";
  Node value = node["foo"];
  node[value] = "foo";
  EXPECT_EQ("value", node["foo"].as<std::string>());
  EXPECT_EQ("foo", node[value].as<std::string>());
  EXPECT_EQ("foo", node["value"].as<std::string>());
  EXPECT_EQ(2, node.size());
}

TEST(NodeTest, SelfReferenceSequence) {
  Node node;
  node[0] = node;
  EXPECT_TRUE(node.IsSequence());
  EXPECT_EQ(1, node.size());
  EXPECT_EQ(node, node[0]);
  EXPECT_EQ(node, node[0][0]);
  EXPECT_EQ(node[0], node[0][0]);
}

TEST(NodeTest, ValueSelfReferenceMap) {
  Node node;
  node["key"] = node;
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ(1, node.size());
  EXPECT_EQ(node, node["key"]);
  EXPECT_EQ(node, node["key"]["key"]);
  EXPECT_EQ(node["key"], node["key"]["key"]);
}

TEST(NodeTest, KeySelfReferenceMap) {
  Node node;
  node[node] = "value";
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ(1, node.size());
  EXPECT_EQ("value", node[node].as<std::string>());
}

TEST(NodeTest, SelfReferenceMap) {
  Node node;
  node[node] = node;
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ(1, node.size());
  EXPECT_EQ(node, node[node]);
  EXPECT_EQ(node, node[node][node]);
  EXPECT_EQ(node[node], node[node][node]);
}

TEST(NodeTest, TempMapVariable) {
  Node node;
  Node tmp = node["key"];
  tmp = "value";
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ(1, node.size());
  EXPECT_EQ("value", node["key"].as<std::string>());
}

TEST(NodeTest, TempMapVariableAlias) {
  Node node;
  Node tmp = node["key"];
  tmp = node["other"];
  node["other"] = "value";
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ(2, node.size());
  EXPECT_EQ("value", node["key"].as<std::string>());
  EXPECT_EQ("value", node["other"].as<std::string>());
  EXPECT_EQ(node["key"], node["other"]);
}

TEST(NodeTest, Bool) {
  Node node;
  node[true] = false;
  EXPECT_TRUE(node.IsMap());
  EXPECT_EQ(false, node[true].as<bool>());
}

TEST(NodeTest, AutoBoolConversion) {
#ifdef _MSC_VER
#pragma warning(disable : 4800)
#endif
  Node node;
  node["foo"] = "bar";
  EXPECT_TRUE(static_cast<bool>(node["foo"]));
  EXPECT_TRUE(!node["monkey"]);
  EXPECT_TRUE(!!node["foo"]);
}

TEST(NodeTest, FloatingPrecisionFloat) {
  const float x = 0.123456789;
  Node node = Node(x);
  EXPECT_EQ(x, node.as<float>());
}

TEST(NodeTest, FloatingPrecisionPositiveInfinityFloat) {
  if (!std::numeric_limits<float>::has_infinity) {
    return;
  }
  const float x = std::numeric_limits<float>::infinity();
  Node node = Node(x);
  EXPECT_EQ(x, node.as<float>());
}

TEST(NodeTest, FloatingPrecisionNegativeInfinityFloat) {
  if (!std::numeric_limits<float>::has_infinity) {
    return;
  }
  const float x = -std::numeric_limits<float>::infinity();
  Node node = Node(x);
  EXPECT_EQ(x, node.as<float>());
}

TEST(NodeTest, FloatingPrecisionNanFloat) {
  if (!std::numeric_limits<float>::has_quiet_NaN) {
    return;
  }
  const float x = std::numeric_limits<float>::quiet_NaN();
  Node node = Node(x);
  EXPECT_TRUE(std::isnan(node.as<float>()));
}

TEST(NodeTest, FloatingPrecisionDouble) {
  const double x = 0.123456789;
  Node node = Node(x);
  EXPECT_EQ(x, node.as<double>());
}

TEST(NodeTest, FloatingPrecisionPositiveInfinityDouble) {
  if (!std::numeric_limits<double>::has_infinity) {
    return;
  }
  const double x = std::numeric_limits<double>::infinity();
  Node node = Node(x);
  EXPECT_EQ(x, node.as<float>());
}

TEST(NodeTest, FloatingPrecisionNegativeInfinityDouble) {
  if (!std::numeric_limits<double>::has_infinity) {
    return;
  }
  const double x = -std::numeric_limits<double>::infinity();
  Node node = Node(x);
  EXPECT_EQ(x, node.as<double>());
}

TEST(NodeTest, FloatingPrecisionNanDouble) {
  if (!std::numeric_limits<double>::has_quiet_NaN) {
    return;
  }
  const double x = std::numeric_limits<double>::quiet_NaN();
  Node node = Node(x);
  EXPECT_TRUE(std::isnan(node.as<double>()));
}

TEST(NodeTest, SpaceChar) {
  Node node = Node(' ');
  EXPECT_EQ(' ', node.as<char>());
}

TEST(NodeTest, CloneNull) {
  Node node;
  Node clone = Clone(node);
  EXPECT_EQ(NodeType::Null, clone.Type());
}

TEST(NodeTest, KeyNodeExitsScope) {
  Node node;
  {
    Node temp("Hello, world");
    node[temp] = 0;
  }
  for (Node::const_iterator it = node.begin(); it != node.end(); ++it) {
    (void)it;
  }
}

TEST(NodeTest, DefaultNodeStyle) {
  Node node;
  EXPECT_EQ(EmitterStyle::Default, node.Style());
}

TEST(NodeTest, AccessNonexistentKeyOnConstNode) {
  YAML::Node node;
  node["3"] = "4";
  const YAML::Node& other = node;
  ASSERT_FALSE(other["5"]);
}

class NodeEmitterTest : public ::testing::Test {
 protected:
  void ExpectOutput(const std::string& output, const Node& node) {
    Emitter emitter;
    emitter << node;
    ASSERT_TRUE(emitter.good());
    EXPECT_EQ(output, emitter.c_str());
  }

  void ExpectAnyOutput(const Node& node, const std::string& output1,
                       const std::string& output2) {
    Emitter emitter;
    emitter << node;
    ASSERT_TRUE(emitter.good());
    EXPECT_THAT(emitter.c_str(), AnyOf(Eq(output1), Eq(output2)));
  }
};

TEST_F(NodeEmitterTest, SimpleFlowSeqNode) {
  Node node;
  node.SetStyle(EmitterStyle::Flow);
  node.push_back(1.5);
  node.push_back(2.25);
  node.push_back(3.125);

  ExpectOutput("[1.5, 2.25, 3.125]", node);
}

TEST_F(NodeEmitterTest, NestFlowSeqNode) {
  Node node, cell0, cell1;

  cell0.push_back(1.5);
  cell0.push_back(2.25);
  cell0.push_back(3.125);

  cell1.push_back(4.5);
  cell1.push_back(5.25);
  cell1.push_back(6.125);

  node.SetStyle(EmitterStyle::Flow);
  node.push_back(cell0);
  node.push_back(cell1);

  ExpectOutput("[[1.5, 2.25, 3.125], [4.5, 5.25, 6.125]]", node);
}

TEST_F(NodeEmitterTest, MixBlockFlowSeqNode) {
  Node node, cell0, cell1;

  cell0.SetStyle(EmitterStyle::Flow);
  cell0.push_back(1.5);
  cell0.push_back(2.25);
  cell0.push_back(3.125);

  cell1.push_back(4.5);
  cell1.push_back(5.25);
  cell1.push_back(6.125);

  node.SetStyle(EmitterStyle::Block);
  node.push_back(cell0);
  node.push_back(cell1);

  ExpectOutput("- [1.5, 2.25, 3.125]\n-\n  - 4.5\n  - 5.25\n  - 6.125", node);
}

TEST_F(NodeEmitterTest, NestBlockFlowMapListNode) {
  Node node, mapNode, blockNode;

  node.push_back(1.5);
  node.push_back(2.25);
  node.push_back(3.125);

  mapNode.SetStyle(EmitterStyle::Flow);
  mapNode["position"] = node;

  blockNode.push_back(1.0625);
  blockNode.push_back(mapNode);

  ExpectOutput("- 1.0625\n- {position: [1.5, 2.25, 3.125]}", blockNode);
}

TEST_F(NodeEmitterTest, NestBlockMixMapListNode) {
  Node node, mapNode, blockNode;

  node.push_back(1.5);
  node.push_back(2.25);
  node.push_back(3.125);

  mapNode.SetStyle(EmitterStyle::Flow);
  mapNode["position"] = node;

  blockNode["scalar"] = 1.0625;
  blockNode["object"] = mapNode;

  ExpectAnyOutput(blockNode,
                  "scalar: 1.0625\nobject: {position: [1.5, 2.25, 3.125]}",
                  "object: {position: [1.5, 2.25, 3.125]}\nscalar: 1.5");
}

TEST_F(NodeEmitterTest, NestBlockMapListNode) {
  Node node, mapNode;

  node.push_back(1.5);
  node.push_back(2.25);
  node.push_back(3.125);

  mapNode.SetStyle(EmitterStyle::Block);
  mapNode["position"] = node;

  ExpectOutput("position:\n  - 1.5\n  - 2.25\n  - 3.125", mapNode);
}

TEST_F(NodeEmitterTest, NestFlowMapListNode) {
  Node node, mapNode;

  node.push_back(1.5);
  node.push_back(2.25);
  node.push_back(3.125);

  mapNode.SetStyle(EmitterStyle::Flow);
  mapNode["position"] = node;

  ExpectOutput("{position: [1.5, 2.25, 3.125]}", mapNode);
}
}
}
