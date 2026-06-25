// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  ConfigTests.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 13.03.25.
//

import Foundation
// Argmax-modification: consolidated the upstream swift-transformers module import(s) (Jinja, Hub) into the single ArgmaxCore module that now vendors this code.
@testable import ArgmaxCore
import Testing


private func createFile(with content: String, encoding: String.Encoding, fileName: String) throws -> URL {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent(fileName)
    guard let data = content.data(using: encoding) else {
        throw NSError(domain: "EncodingError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Could not encode string with \(encoding)"])
    }
    try data.write(to: fileURL)
    return fileURL
}

@Suite("Hub Config Tests")
struct ConfigTests {
    @Test("Data hashing produces different hashes for unequal values")
    func hashable() throws {
        let testCases: [(Config.Data, Config.Data)] = [
            (.integer(1), .integer(2)),
            (.string("a"), .string("2")),
            (.boolean(true), .string("T")),
            (.boolean(true), .boolean(false)),
            (.floating(1.1), .floating(1.1000001)),
            (.token((1, "a")), .token((1, "b"))),
            (.token((1, "a")), .token((2, "a"))),
            (.dictionary(["1": Config()]), .dictionary(["1": 1])),
            (.dictionary(["1": 10]), .dictionary(["2": 10])),
            (.array(["1", "2"]), .array(["1", "3"])),
            (.array([1, 2]), .array([2, 1])),
            (.array([true, false]), .array([true, true])),
        ]

        for (lhs, rhs) in testCases {
            var leftHasher = Hasher()
            var rightHasher = Hasher()
            lhs.hash(into: &leftHasher)
            rhs.hash(into: &rightHasher)
            #expect(leftHasher.finalize() != rightHasher.finalize())
        }
    }

    @Test("ExpressibleByStringLiteral")
    func stringLiteral() throws {
        let cfg: Config = "test"
        #expect(cfg == "test")
    }

    @Test("ExpressibleByIntegerLiteral")
    func integerLiteral() throws {
        let cfg: Config = 678
        #expect(cfg == 678)
    }

    @Test("ExpressibleByBooleanLiteral")
    func booleanLiteral() throws {
        let cfg: Config = true
        #expect(cfg == true)
    }

    @Test("ExpressibleByFloatLiteral")
    func floatLiteral() throws {
        let cfg: Config = 1.1
        #expect(cfg == 1.1)
    }

    @Test("ExpressibleByDictionaryLiteral")
    func dictionaryLiteral() throws {
        let cfg: Config = ["key": 1.1]
        #expect(cfg["key"].floating(or: 0) == 1.1)
    }

    @Test("ExpressibleByArrayLiteral")
    func arrayLiteral() throws {
        let cfg: Config = [1.1, 1.2]
        #expect(cfg[0] == 1.1)
        #expect(cfg[1] == 1.2)
    }

    @Test("Key and index subscripts")
    func accessors_keyAndIndex() throws {
        let cfgKey: Config = ["key": 1.1]
        #expect(cfgKey["key"] == 1.1)
        #expect(cfgKey["non_existent"].isNull())
        #expect(cfgKey[1].isNull())

        let cfgIndex: Config = [1, 2, 3, 4]
        #expect(cfgIndex[1] == 2)
        #expect(cfgIndex[99].isNull())
        #expect(cfgIndex[-1].isNull())
    }

    @Test("Dynamic member lookup for snake_case and camelCase")
    func dynamicLookup() throws {
        let cfg: Config = ["model_type": "bert"]
        #expect(cfg["model_type"] == "bert")
        #expect(cfg.modelType == "bert")
        #expect(cfg.model_type == "bert")
        #expect(cfg.unknown_key.isNull())
    }

    @Test("Array and dictionary conversions")
    func arrayAndDictionary() throws {
        let cfgArray: Config = [1, 2, 3, 4]
        #expect(cfgArray.array() == [1, 2, 3, 4])
        #expect(cfgArray.get() == [1, 2, 3, 4])
        #expect(cfgArray.get(or: []) == [1, 2, 3, 4])
        #expect(cfgArray["fake_key"].isNull())
        #expect(cfgArray.dictionary() == nil)
        #expect(cfgArray.dictionary(or: ["a": 1]) == ["a": 1])

        let cfgDict: Config = ["a": 1, "b": 2, "c": 3, "d": 4]
        #expect(cfgDict.dictionary() == ["a": 1, "b": 2, "c": 3, "d": 4])
        #expect(cfgDict.get() == ["a": 1, "b": 2, "c": 3, "d": 4])
        #expect(cfgDict.get(or: [:]) == ["a": 1, "b": 2, "c": 3, "d": 4])
        #expect(cfgDict[666].isNull())
        #expect(cfgDict.array() == nil)
        #expect(cfgDict.array(or: ["a"]) == ["a"])
    }

    @Test("Arrays of strings and Configs")
    func arraysOfStringsAndConfigs() throws {
        let cfgStrings: Config = ["a", "b", "c"]
        #expect(cfgStrings.array() == ["a", "b", "c"])
        #expect(cfgStrings.get() == ["a", "b", "c"])
        #expect(cfgStrings.get() == [BinaryDistinctString("a"), BinaryDistinctString("b"), BinaryDistinctString("c")])
        #expect(cfgStrings.get(or: []) == [BinaryDistinctString("a"), BinaryDistinctString("b"), BinaryDistinctString("c")])
        #expect(cfgStrings.get(or: []) == ["a", "b", "c"])
        #expect(cfgStrings.dictionary() == nil)
        #expect(cfgStrings.dictionary(or: ["a": 1]) == ["a": 1])

        let cfgConfigs: Config = [Config("a"), Config("b")]
        #expect(cfgConfigs.array() == ["a", "b"])
        #expect(cfgConfigs.get() == ["a", "b"])
        #expect(cfgConfigs.get() == [BinaryDistinctString("a"), BinaryDistinctString("b")])
        #expect(cfgConfigs.get(or: []) == [BinaryDistinctString("a"), BinaryDistinctString("b")])
        #expect(cfgConfigs.get(or: []) == ["a", "b"])
        #expect(cfgConfigs.dictionary() == nil)
        #expect(cfgConfigs.dictionary(or: ["a": 1]) == ["a": 1])
    }

    @Test("Dictionary of Configs coerces to expected map")
    func dictionaryOfConfigs() throws {
        let cfg: Config = ["a": .init([1, 2]), "b": .init([3, 4])]
        let exp = [BinaryDistinctString("a"): Config([1, 2]), BinaryDistinctString("b"): Config([3, 4])]
        #expect(cfg.dictionary() == exp)
        #expect(cfg.get() == exp)
        #expect(cfg.get(or: [:]) == exp)
        #expect(cfg[666].isNull())
        #expect(cfg.array() == nil)
        #expect(cfg.array(or: ["a"]) == ["a"])
    }

    @Test("Codable round-trip retains values")
    func codableRoundTrip() throws {
        let cfg: Config = [
            "dict_of_floats": ["key1": 1.1],
            "dict_of_ints": ["key2": 100],
            "dict_of_strings": ["key3": "abc"],
            "dict_of_bools": ["key4": false],
            "dict_of_dicts": ["key5": ["key_inside": 99]],
            "dict_of_tokens": ["key6": .init((12, "dfe"))],
            "arr_empty": [],
            "arr_of_ints": [1, 2, 3],
            "arr_of_floats": [1.1, 1.2],
            "arr_of_strings": ["a", "b"],
            "arr_of_bools": [true, false],
            "arr_of_dicts": [["key7": 1.1], ["key8": 1.2]],
            "arr_of_tokens": [.init((1, "a")), .init((2, "b"))],
            "int": 678,
            "float": 1.1,
            "string": "test",
            "bool": true,
            "token": .init((1, "test")),
            "null": Config(),
        ]

        let data = try JSONEncoder().encode(cfg)
        let got = try JSONDecoder().decode(Config.self, from: data)

        #expect(got == cfg)
        #expect(got["dict_of_floats"]["key1"] == 1.1)
        #expect(got["dict_of_ints"]["key2"] == 100)
        #expect(got["dict_of_strings"]["key3"] == "abc")
        #expect(got["dict_of_bools"]["key4"] == false)
        #expect(got["dict_of_dicts"]["key5"]["key_inside"] == 99)
        #expect(got["dict_of_tokens"]["key6"].token()?.0 == 12)
        #expect(got["dict_of_tokens"]["key6"].token()?.1 == "dfe")
        #expect(got["arr_empty"].array()?.count == 0)
        #expect(got["arr_of_ints"] == [1, 2, 3])
        #expect(got["arr_of_floats"] == [1.1, 1.2])
        #expect(got["arr_of_strings"] == ["a", "b"])
        #expect(got["arr_of_bools"] == [true, false])
        #expect(got["arr_of_dicts"][1]["key8"] == 1.2)
        #expect(got["arr_of_tokens"][1].token(or: (0, "")) == (2, "b"))
        #expect(got["arr_of_tokens"][2].token() == nil)
        #expect(got["int"] == 678)
        #expect(got["float"] == 1.1)
        #expect(got["string"] == "test")
        #expect(got["bool"] == true)
        #expect(got["token"].token(or: (0, "")) == (1, "test"))
        #expect(got["null"].isNull())
    }

    @Test("Equatable: string, integer, float, boolean, token, dictionary")
    func equatableConversions() throws {
        // String
        do {
            let cfg = Config("a")
            #expect(cfg == "a")
            #expect(cfg.get() == "a")
            #expect(cfg.get(or: "b") == "a")
            #expect(cfg.string() == "a")
            #expect(cfg.string(or: "b") == "a")
            #expect(cfg.get() == BinaryDistinctString("a"))
            // Argmax-modification: typed the `or:` default as BinaryDistinctString. Upstream wrote
            // `get(or: "b")` and relied on the RHS to select the BinaryDistinctString overload, but
            // under this toolchain the bare string literal resolves to the String `get(or:)` overload,
            // comparing a String to a BinaryDistinctString. Explicit typing restores the intended overload.
            #expect(cfg.get(or: BinaryDistinctString("b")) == BinaryDistinctString("a"))
            #expect(cfg.binaryDistinctString() == "a")
            #expect(cfg.binaryDistinctString(or: "b") == "a")
        }

        // Integer
        do {
            let cfg = Config(1)
            #expect(cfg == 1)
            #expect(cfg.get() == 1)
            #expect(cfg.get(or: 2) == 1)
            #expect(cfg.integer() == 1)
            #expect(cfg.integer(or: 2) == 1)
        }

        // Floating
        do {
            let testCases: [(Config, Float)] = [
                (Config(1.1), 1.1),
                (Config(1), 1.0),
            ]
            for (cfg, exp) in testCases {
                #expect(cfg == .init(exp))
                #expect(cfg.get() == exp)
                #expect(cfg.get(or: 2.2) == exp)
                #expect(cfg.floating() == exp)
                #expect(cfg.floating(or: 2.2) == exp)
            }
        }

        // Boolean
        do {
            let testCases: [(Config, Bool)] = [
                (Config(true), true),
                (Config(1), true),
                (Config("T"), true),
                (Config("t"), true),
                (Config("TRUE"), true),
                (Config("True"), true),
                (Config("true"), true),
                (Config("F"), false),
                (Config("f"), false),
                (Config("FALSE"), false),
                (Config("False"), false),
                (Config("false"), false),
            ]
            for (cfg, exp) in testCases {
                #expect(cfg.get() == exp)
                #expect(cfg.get(or: !exp) == exp)
                #expect(cfg.boolean() == exp)
                #expect(cfg.boolean(or: !exp) == exp)
            }
        }

        // Token
        do {
            let cfg = Config((1, "a"))
            let exp: (UInt, String) = (1, "a")
            #expect(cfg == .init((1, "a")))
            #expect(cfg.get()! == exp)
            #expect(cfg.get(or: (2, "b")) == exp)
            #expect(cfg.token()! == exp)
            #expect(cfg.token(or: (2, "b")) == exp)
        }

        // Dictionary-like
        do {
            let testCases: [(Config, Int)] = [
                (Config(["a": 1]), 1),
                (Config(["a": 2] as [NSString: Any]), 2),
                (Config(["a": 3] as [NSString: Config]), 3),
                (Config([BinaryDistinctString("a"): 4] as [BinaryDistinctString: Config]), 4),
                (Config(["a": Config(5)]), 5),
                (Config(["a": 6]), 6),
                (Config((BinaryDistinctString("a"), 7)), 7),
            ]
            for (cfg, exp) in testCases {
                #expect(cfg["a"] == Config(exp))
                #expect(cfg.get(or: [:])["a"] == Config(exp))
            }
        }
    }

    @Test("JSON decoding supports UTF-8, UTF-16LE, and UTF-16BE")
    func textEncoding_utf16Variants() throws {
        let json = """
                {
                  "a": ["val_1", "val_2"],
                  "b": 2,
                  "c": [[10, "tkn_1"], [12, "tkn_2"], [4, "tkn_3"]],
                  "d": false,
                  "e": {
                    "e_1": 1.1,
                    "e_2": [1, 2, 3]
                  },
                  "f": null
                }
            """

        let urlUTF8 = try createFile(with: json, encoding: .utf8, fileName: "config_utf8.json")
        let urlUTF16LE = try createFile(with: json, encoding: .utf16LittleEndian, fileName: "config_utf16_le.json")
        let urlUTF16BE = try createFile(with: json, encoding: .utf16BigEndian, fileName: "config_utf16_be.json")

        let dataUTF8 = try Data(contentsOf: urlUTF8)
        let dataUTF16LE = try Data(contentsOf: urlUTF16LE)
        let dataUTF16BE = try Data(contentsOf: urlUTF16BE)

        #expect(dataUTF8.count != dataUTF16LE.count)
        #expect(dataUTF8.count != dataUTF16BE.count)

        let decoder = JSONDecoder()
        let configUTF8 = try decoder.decode(Config.self, from: dataUTF8)
        let configUTF16LE = try decoder.decode(Config.self, from: dataUTF16LE)
        let configUTF16BE = try decoder.decode(Config.self, from: dataUTF16BE)

        #expect(configUTF8 == configUTF16LE)
        #expect(configUTF8 == configUTF16BE)

        try FileManager.default.removeItem(at: urlUTF8)
        try FileManager.default.removeItem(at: urlUTF16LE)
        try FileManager.default.removeItem(at: urlUTF16BE)
    }

    @Test("Unicode keys remain distinct (\"à\" vs \"à\")")
    func unicodeKeysAreDistinct() throws {
        let json = "{\"vocab\": {\"à\": 1, \"à\": 2}}"
        let data = json.data(using: .utf8)
        let dict = try JSONSerialization.jsonObject(with: data!, options: []) as! [NSString: Any]
        let config = Config(dict)
        let vocab = config["vocab"].dictionary(or: [:])
        #expect(vocab.count == 2)
    }

    @Test("Token tuple extraction from array and JSON")
    func tokenValueExtraction() throws {
        let config1 = Config(["cls": ["str" as String, 100 as UInt] as [Any]])
        let tokenValue1 = config1.cls?.token()
        #expect(tokenValue1?.0 == 100)
        #expect(tokenValue1?.1 == "str")

        let data = #"{"cls": ["str", 100]}"#.data(using: .utf8)!
        let dict = try JSONSerialization.jsonObject(with: data, options: []) as! [NSString: Any]
        let config2 = Config(dict)
        let tokenValue2 = config2.cls?.token()
        #expect(tokenValue2?.0 == 100)
        #expect(tokenValue2?.1 == "str")
    }

    // Argmax-modification: removed the `templating_completeExample` test.
}
