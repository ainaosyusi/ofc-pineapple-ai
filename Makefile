# OFC Pineapple AI - Makefile

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I./src/cpp

# ソースファイル
SRC_DIR = src/cpp
TEST_DIR = tests/cpp
BUILD_DIR = build

# テスト実行ファイル
TEST_TARGET = $(BUILD_DIR)/test_ofc

# デフォルトターゲット
all: test

# ビルドディレクトリ作成
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# テストビルド
test: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $(TEST_TARGET) $(TEST_DIR)/test_main.cpp
	./$(TEST_TARGET)

# デバッグビルド
debug: CXXFLAGS = -std=c++17 -g -O0 -Wall -Wextra -I./src/cpp
debug: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/test_ofc_debug $(TEST_DIR)/test_main.cpp

# ベンチマーク
benchmark: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DNDEBUG -march=native -o $(BUILD_DIR)/bench_ofc $(TEST_DIR)/test_main.cpp
	./$(BUILD_DIR)/bench_ofc

# クリーン
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all test debug benchmark clean
