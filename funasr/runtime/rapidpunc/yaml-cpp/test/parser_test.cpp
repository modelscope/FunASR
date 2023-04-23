#include <yaml-cpp/depthguard.h>
#include "yaml-cpp/parser.h"
#include "yaml-cpp/exceptions.h"
#include "mock_event_handler.h"
#include "gtest/gtest.h"

using YAML::Parser;
using YAML::MockEventHandler;
using ::testing::NiceMock;
using ::testing::StrictMock;

TEST(ParserTest, Empty) {
    Parser parser;

    EXPECT_FALSE(parser);

    StrictMock<MockEventHandler> handler;
    EXPECT_FALSE(parser.HandleNextDocument(handler));
}

TEST(ParserTest, CVE_2017_5950) {
    std::string excessive_recursion;
    for (auto i = 0; i != 16384; ++i)
        excessive_recursion.push_back('[');
    std::istringstream input{excessive_recursion};
    Parser parser{input};

    NiceMock<MockEventHandler> handler;
    EXPECT_THROW(parser.HandleNextDocument(handler), YAML::DeepRecursion);
}

TEST(ParserTest, CVE_2018_20573) {
    std::string excessive_recursion;
    for (auto i = 0; i != 20535; ++i)
        excessive_recursion.push_back('{');
    std::istringstream input{excessive_recursion};
    Parser parser{input};

    NiceMock<MockEventHandler> handler;
    EXPECT_THROW(parser.HandleNextDocument(handler), YAML::DeepRecursion);
}

TEST(ParserTest, CVE_2018_20574) {
    std::string excessive_recursion;
    for (auto i = 0; i != 21989; ++i)
        excessive_recursion.push_back('{');
    std::istringstream input{excessive_recursion};
    Parser parser{input};

    NiceMock<MockEventHandler> handler;
    EXPECT_THROW(parser.HandleNextDocument(handler), YAML::DeepRecursion);
}

TEST(ParserTest, CVE_2019_6285) {
    std::string excessive_recursion;
    for (auto i = 0; i != 23100; ++i)
        excessive_recursion.push_back('[');
    excessive_recursion.push_back('f');
    std::istringstream input{excessive_recursion};
    Parser parser{input};

    NiceMock<MockEventHandler> handler;
    EXPECT_THROW(parser.HandleNextDocument(handler), YAML::DeepRecursion);
}
