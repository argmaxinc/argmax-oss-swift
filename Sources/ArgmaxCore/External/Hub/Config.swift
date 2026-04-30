// Originally from: https://github.com/huggingface/swift-transformers
// Version: 1.1.6 (commit: 573e5c9036c2f136b3a8a071da8e8907322403d0)
// License: Apache 2.0 (https://github.com/huggingface/swift-transformers/blob/main/LICENSE)
// Copyright 2022 Hugging Face SAS
// Modified by Argmax, Inc. See Argmax-modification: comments for changes.
//

//
//  Config.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 06.03.25.

import Foundation
// Argmax-modification: removed `import Jinja` — Jinja dependency
// Argmax-modification: Config made internal — only used within ArgmaxCore

// MARK: - Configuration files with dynamic lookup

/// A flexible configuration structure for handling JSON-like data with dynamic member lookup.
///
/// Config provides a type-safe way to work with configuration files from the Hugging Face Hub,
/// supporting multiple data types and automatic type conversion. It uses dynamic member lookup
/// to provide convenient access to nested configuration values while maintaining type safety
/// through explicit conversion methods.
// Argmax-modification: removed public — Config is internal
@dynamicMemberLookup
struct Config: Hashable, Sendable,
    ExpressibleByStringLiteral,
    ExpressibleByIntegerLiteral,
    ExpressibleByBooleanLiteral,
    ExpressibleByFloatLiteral,
    ExpressibleByDictionaryLiteral,
    ExpressibleByArrayLiteral,
    ExpressibleByExtendedGraphemeClusterLiteral,
    CustomStringConvertible
{
    // Argmax-modification: removed public from typealiases, nested types, inits, methods, subscripts, and extensions — Config is internal
    /// Type alias for configuration keys using binary-distinct strings.
    typealias Key = BinaryDistinctString
    /// Type alias for configuration values.
    typealias Value = Config

    private let value: Data

    /// The underlying data types supported by the configuration system.
    ///
    /// This enumeration represents all possible value types that can be stored
    /// in a configuration, providing type-safe access to different data formats.
    // Argmax-modification: removed public — Data is internal
    enum Data: Sendable {
        /// Represents a null/nil value.
        case null
        /// A string value stored as a binary-distinct string.
        case string(BinaryDistinctString)
        /// An integer numeric value.
        case integer(Int)
        /// A boolean true/false value.
        case boolean(Bool)
        /// A floating-point numeric value.
        case floating(Float)
        /// A dictionary mapping keys to configuration values.
        case dictionary([BinaryDistinctString: Config])
        /// An array of configuration values.
        case array([Config])
        /// A token tuple containing an ID and string value.
        case token((UInt, BinaryDistinctString))

        static func == (lhs: Data, rhs: Data) -> Bool {
            switch (lhs, rhs) {
            case (.null, .null):
                return true
            case let (.string(lhs), _):
                if let rhs = rhs.string() {
                    return lhs == BinaryDistinctString(rhs)
                }
            case let (.integer(lhs), _):
                if let rhs = rhs.integer() {
                    return lhs == rhs
                }
            case let (.boolean(lhs), _):
                if let rhs = rhs.boolean() {
                    return lhs == rhs
                }
            case let (.floating(lhs), _):
                if let rhs = rhs.floating() {
                    return lhs == rhs
                }
            case let (.dictionary(lhs), .dictionary(rhs)):
                return lhs == rhs
            case let (.array(lhs), .array(rhs)):
                return lhs == rhs
            case let (.token(lhs), .token(rhs)):
                return lhs == rhs
            default:
                return false
            }

            // right hand side might be a super set of left hand side
            switch rhs {
            case let .string(rhs):
                if let lhs = lhs.string() {
                    return BinaryDistinctString(lhs) == rhs
                }
            case let .integer(rhs):
                if let lhs = lhs.integer() {
                    return lhs == rhs
                }
            case let .boolean(rhs):
                if let lhs = lhs.boolean() {
                    return lhs == rhs
                }
            case let .floating(rhs):
                if let lhs = lhs.floating() {
                    return lhs == rhs
                }
            default:
                return false
            }

            return false
        }

        var description: String {
            switch self {
            case .null:
                "null"
            case let .string(value):
                "\"\(value)\""
            case let .integer(value):
                "\(value)"
            case let .boolean(value):
                "\(value)"
            case let .floating(value):
                "\(value)"
            case let .array(arr):
                "[\(arr)]"
            case let .dictionary(val):
                "{\(val)}"
            case let .token(val):
                "(\(val.0), \(val.1))"
            }
        }

        func string() -> String? {
            if case let .string(val) = self {
                return val.string
            }
            return nil
        }

        func boolean() -> Bool? {
            if case let .boolean(val) = self {
                return val
            }
            if case let .integer(val) = self {
                return val == 1
            }
            if case let .string(val) = self {
                switch val.string.lowercased() {
                case "true", "t", "1":
                    return true
                case "false", "f", "0":
                    return false
                default:
                    return nil
                }
            }
            return nil
        }

        func integer() -> Int? {
            if case let .integer(val) = self {
                return val
            }
            return nil
        }

        func floating() -> Float? {
            if case let .floating(val) = self {
                return val
            }
            if case let .integer(val) = self {
                return Float(val)
            }
            return nil
        }
    }

    init() {
        self.value = .null
    }

    init(_ value: BinaryDistinctString) {
        self.value = .string(value)
    }

    init(_ value: String) {
        self.init(stringLiteral: value)
    }

    init(_ value: Int) {
        self.init(integerLiteral: value)
    }

    init(_ value: Bool) {
        self.init(booleanLiteral: value)
    }

    init(_ value: Float) {
        self.init(floatLiteral: value)
    }

    init(_ value: [Config]) {
        self.value = .array(value)
    }

    init(_ values: (BinaryDistinctString, Config)...) {
        var dict = [BinaryDistinctString: Config]()
        for (key, value) in values {
            dict[key] = value
        }
        self.value = .dictionary(dict)
    }

    init(_ value: [BinaryDistinctString: Config]) {
        self.value = .dictionary(value)
    }

    init(_ dictionary: [NSString: Any]) {
        self.value = Config.convertToBinaryDistinctKeys(dictionary as Any).value
    }

    init(_ dictionary: [String: Config]) {
        self.value = Config.convertToBinaryDistinctKeys(dictionary as Any).value
    }

    init(_ dictionary: [NSString: Config]) {
        self.value = Config.convertToBinaryDistinctKeys(dictionary as Any).value
    }

    init(_ token: (UInt, BinaryDistinctString)) {
        self.value = .token(token)
    }

    private static func convertToBinaryDistinctKeys(_ object: Any) -> Config {
        if let dict = object as? [NSString: Any] {
            Config(Dictionary(uniqueKeysWithValues: dict.map { (BinaryDistinctString($0.key), convertToBinaryDistinctKeys($0.value)) }))
        } else if let array = object as? [Any] {
            Config(array.map { convertToBinaryDistinctKeys($0) })
        } else {
            switch object {
            case let obj as String:
                Config(obj)
            case let obj as Int:
                Config(obj)
            case let obj as Float:
                Config(obj)
            case let obj as Bool:
                Config(obj)
            case let obj as NSNumber:
                if CFNumberIsFloatType(obj) {
                    Config(obj.floatValue)
                } else {
                    Config(obj.intValue)
                }
            case _ as NSNull:
                Config()
            case let obj as Config:
                obj
            case let obj as (UInt, String):
                Config((obj.0, BinaryDistinctString(obj.1)))
            default:
                fatalError("unknown type: \(type(of: object)) \(object)")
            }
        }
    }

    // MARK: constructors

    /// Conformance to ExpressibleByStringLiteral
    init(stringLiteral value: String) {
        self.value = .string(.init(value))
    }

    /// Conformance to ExpressibleByIntegerLiteral
    init(integerLiteral value: Int) {
        self.value = .integer(value)
    }

    /// Conformance to ExpressibleByBooleanLiteral
    init(booleanLiteral value: Bool) {
        self.value = .boolean(value)
    }

    /// Conformance to ExpressibleByFloatLiteral
    init(floatLiteral value: Float) {
        self.value = .floating(value)
    }

    init(dictionaryLiteral elements: (BinaryDistinctString, Config)...) {
        let dict = elements.reduce(into: [BinaryDistinctString: Config]()) { result, element in
            result[element.0] = element.1
        }

        self.value = .dictionary(dict)
    }

    init(arrayLiteral elements: Config...) {
        self.value = .array(elements)
    }

    func isNull() -> Bool {
        if case .null = self.value {
            return true
        }
        return false
    }

    // MARK: getters - string

    func get() -> String? {
        self.string()
    }

    func get(or: String) -> String? {
        self.string(or: or)
    }

    func string() -> String? {
        self.value.string()
    }

    func string(or: String) -> String {
        if let val: String = self.string() {
            return val
        }
        return or
    }

    func get() -> BinaryDistinctString? {
        self.binaryDistinctString()
    }

    func get(or: BinaryDistinctString) -> BinaryDistinctString? {
        self.binaryDistinctString(or: or)
    }

    func binaryDistinctString() -> BinaryDistinctString? {
        if case let .string(val) = self.value {
            return val
        }
        return nil
    }

    func binaryDistinctString(or: BinaryDistinctString) -> BinaryDistinctString {
        if let val: BinaryDistinctString = self.binaryDistinctString() {
            return val
        }
        return or
    }

    // MARK: getters - boolean

    func get() -> Bool? {
        self.boolean()
    }

    func get(or: Bool) -> Bool? {
        self.boolean(or: or)
    }

    func boolean() -> Bool? {
        self.value.boolean()
    }

    func boolean(or: Bool) -> Bool {
        if let val = self.boolean() {
            return val
        }
        return or
    }

    // MARK: getters - integer

    func get() -> Int? {
        self.integer()
    }

    func get(or: Int) -> Int? {
        self.integer(or: or)
    }

    func integer() -> Int? {
        self.value.integer()
    }

    func integer(or: Int) -> Int {
        if let val = self.integer() {
            return val
        }
        return or
    }

    // MARK: getters/operators - floating

    func get() -> Float? {
        self.value.floating()
    }

    func get(or: Float) -> Float? {
        self.floating(or: or)
    }

    func floating() -> Float? {
        self.value.floating()
    }

    func floating(or: Float) -> Float {
        if let val = self.value.floating() {
            return val
        }
        return or
    }

    // MARK: getters - dictionary

    func get() -> [BinaryDistinctString: Int]? {
        if let dict = self.dictionary() {
            return dict.reduce(into: [:]) { result, element in
                if let val = element.value.value.integer() {
                    result[element.key] = val
                }
            }
        }

        return nil
    }

    func get() -> [BinaryDistinctString: Config]? {
        self.dictionary()
    }

    func get(or: [BinaryDistinctString: Config]) -> [BinaryDistinctString: Config] {
        self.dictionary(or: or)
    }

    // Argmax-modification: removed jinjaValue() - Jinja dependency

    func dictionary() -> [BinaryDistinctString: Config]? {
        if case let .dictionary(val) = self.value {
            return val
        }
        return nil
    }

    func dictionary(or: [BinaryDistinctString: Config]) -> [BinaryDistinctString: Config] {
        if let val = self.dictionary() {
            return val
        }
        return or
    }

    // MARK: getters - array

    func get() -> [String]? {
        if let arr = self.array() {
            return arr.reduce(into: []) { result, element in
                if let val: String = element.value.string() {
                    result.append(val)
                }
            }
        }

        return nil
    }

    func get(or: [String]) -> [String] {
        if let arr: [String] = self.get() {
            return arr
        }

        return or
    }

    func get() -> [BinaryDistinctString]? {
        if let arr = self.array() {
            return arr.reduce(into: []) { result, element in
                if let val: BinaryDistinctString = element.binaryDistinctString() {
                    result.append(val)
                }
            }
        }

        return nil
    }

    func get(or: [BinaryDistinctString]) -> [BinaryDistinctString] {
        if let arr: [BinaryDistinctString] = self.get() {
            return arr
        }

        return or
    }

    func get() -> [Config]? {
        self.array()
    }

    func get(or: [Config]) -> [Config] {
        self.array(or: or)
    }

    func array() -> [Config]? {
        if case let .array(val) = self.value {
            return val
        }
        return nil
    }

    func array(or: [Config]) -> [Config] {
        if let val = self.array() {
            return val
        }
        return or
    }

    // MARK: getters - token

    func get() -> (UInt, String)? {
        self.token()
    }

    func get(or: (UInt, String)) -> (UInt, String) {
        self.token(or: or)
    }

    func token() -> (UInt, String)? {
        if case let .token(val) = self.value {
            return (val.0, val.1.string)
        }

        if case let .array(arr) = self.value {
            guard arr.count == 2 else {
                return nil
            }
            guard let token = arr[0].string() else {
                return nil
            }
            guard let id = arr[1].integer() else {
                return nil
            }

            return (UInt(id), token)
        }

        return nil
    }

    func token(or: (UInt, String)) -> (UInt, String) {
        if let val = self.token() {
            return val
        }
        return or
    }

    // MARK: subscript

    subscript(index: BinaryDistinctString) -> Config {
        if let dict = self.dictionary() {
            return dict[index] ?? dict[self.uncamelCase(index)] ?? Config()
        }

        return Config()
    }

    subscript(index: Int) -> Config {
        if let arr = self.array(), index >= 0, index < arr.count {
            return arr[index]
        }

        return Config()
    }

    subscript(dynamicMember member: String) -> Config? {
        if let dict = self.dictionary() {
            return dict[BinaryDistinctString(member)] ?? dict[self.uncamelCase(BinaryDistinctString(member))] ?? Config()
        }

        return nil // backward compatibility
    }

    subscript(dynamicMember member: String) -> Config {
        if let dict = self.dictionary() {
            return dict[BinaryDistinctString(member)] ?? dict[self.uncamelCase(BinaryDistinctString(member))] ?? Config()
        }

        return Config()
    }

    func uncamelCase(_ string: BinaryDistinctString) -> BinaryDistinctString {
        let scalars = string.string.unicodeScalars
        var result = ""

        var previousCharacterIsLowercase = false
        for scalar in scalars {
            if CharacterSet.uppercaseLetters.contains(scalar) {
                if previousCharacterIsLowercase {
                    result += "_"
                }
                let lowercaseChar = Character(scalar).lowercased()
                result += lowercaseChar
                previousCharacterIsLowercase = false
            } else {
                result += String(scalar)
                previousCharacterIsLowercase = true
            }
        }

        return BinaryDistinctString(result)
    }

    var description: String {
        "\(self.value.description)"
    }
}

/// Old style, deprecated getters
// Argmax-modification: removed public — Config is internal
extension Config {
    @available(*, deprecated, message: "Use string() instead")
    var stringValue: String? { string() }

    @available(*, deprecated, message: "Use integer() instead")
    var intValue: Int? { integer() }

    @available(*, deprecated, message: "Use boolean() instead")
    var boolValue: Bool? { boolean() }

    @available(*, deprecated, message: "Use array() instead")
    var arrayValue: [Config]? { array() }

    @available(*, deprecated, message: "Use token() instead")
    var tokenValue: (UInt, String)? { token() }
}

// Argmax-modification: removed public from init(from:) and encode(to:); Decoder/Encoder qualified as Swift.Decoder/Swift.Encoder — disambiguates from internal protocol Decoder
extension Config: Codable {
    init(from decoder: Swift.Decoder) throws {
        // Try decoding as a single value first (for scalars and null)
        let singleValueContainer = try? decoder.singleValueContainer()
        if let container = singleValueContainer {
            if container.decodeNil() {
                self.value = .null
                return
            }
            do {
                let intValue = try container.decode(Int.self)
                self.value = .integer(intValue)
                return
            } catch {}
            do {
                let floatValue = try container.decode(Float.self)
                self.value = .floating(floatValue)
                return
            } catch {}
            do {
                let boolValue = try container.decode(Bool.self)
                self.value = .boolean(boolValue)
                return
            } catch {}
            do {
                let stringValue = try container.decode(String.self)
                self.value = .string(.init(stringValue))
                return
            } catch {}
        }

        if let tupple = Self.decodeTuple(decoder) {
            self.value = tupple
            return
        }
        if let array = Self.decodeArray(decoder) {
            self.value = array
            return
        }

        if let dict = Self.decodeDictionary(decoder) {
            self.value = dict
            return
        }

        self.value = .null
    }

    private static func decodeTuple(_ decoder: Swift.Decoder) -> Data? {
        let unkeyedContainer = try? decoder.unkeyedContainer()
        if var container = unkeyedContainer {
            if container.count == 2 {
                do {
                    let intValue = try container.decode(UInt.self)
                    let stringValue = try container.decode(String.self)
                    return .token((intValue, .init(stringValue)))
                } catch {}
            }
        }
        return nil
    }

    private static func decodeArray(_ decoder: Swift.Decoder) -> Data? {
        do {
            if var container = try? decoder.unkeyedContainer() {
                var elements: [Config] = []
                while !container.isAtEnd {
                    let element = try container.decode(Config.self)
                    elements.append(element)
                }
                return .array(elements)
            }
        } catch {}
        return nil
    }

    private static func decodeDictionary(_ decoder: Swift.Decoder) -> Data? {
        do {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            var dictionaryValues: [BinaryDistinctString: Config] = [:]
            for key in container.allKeys {
                let value = try container.decode(Config.self, forKey: key)
                dictionaryValues[BinaryDistinctString(key.stringValue)] = value
            }

            return .dictionary(dictionaryValues)
        } catch {
            return nil
        }
    }

    func encode(to encoder: Swift.Encoder) throws {
        switch self.value {
        case .null:
            var container = encoder.singleValueContainer()
            try container.encodeNil()
        case let .integer(val):
            var container = encoder.singleValueContainer()
            try container.encode(val)
        case let .floating(val):
            var container = encoder.singleValueContainer()
            try container.encode(val)
        case let .boolean(val):
            var container = encoder.singleValueContainer()
            try container.encode(val)
        case let .string(val):
            var container = encoder.singleValueContainer()
            try container.encode(val.string)
        case let .dictionary(val):
            var container = encoder.container(keyedBy: CodingKeys.self)
            for (key, value) in val {
                try container.encode(value, forKey: CodingKeys(stringValue: key.string)!)
            }
        case let .array(val):
            var container = encoder.unkeyedContainer()
            try container.encode(contentsOf: val)
        case let .token(val):
            var tupple = encoder.unkeyedContainer()
            try tupple.encode(val.0)
            try tupple.encode(val.1.string)
        }
    }

    private struct CodingKeys: CodingKey {
        var stringValue: String
        init?(stringValue: String) {
            self.stringValue = stringValue
        }

        var intValue: Int? { nil }
        init?(intValue: Int) { nil }
    }
}

// Argmax-modification: removed public from == — Config is internal
extension Config: Equatable {
    static func == (lhs: Config, rhs: Config) -> Bool {
        lhs.value == rhs.value
    }
}

// Argmax-modification: removed public from hash(into:) — Config is internal
extension Config.Data: Hashable {
    func hash(into hasher: inout Hasher) {
        switch self {
        case .null:
            hasher.combine(0) // Discriminator for null
        case let .string(s):
            hasher.combine(1) // Discriminator for string
            hasher.combine(s)
        case let .integer(i):
            hasher.combine(2) // Discriminator for integer
            hasher.combine(i)
        case let .boolean(b):
            hasher.combine(3) // Discriminator for boolean
            hasher.combine(b)
        case let .floating(f):
            hasher.combine(4) // Discriminator for floating
            hasher.combine(f)
        case let .dictionary(d):
            hasher.combine(5) // Discriminator for dict
            d.hash(into: &hasher)
        case let .array(a):
            hasher.combine(6) // Discriminator for array
            for e in a {
                e.hash(into: &hasher)
            }
        case let .token(a):
            hasher.combine(7) // Discriminator for token
            a.0.hash(into: &hasher)
            a.1.hash(into: &hasher)
        }
    }
}

// Argmax-modification: removed ConfigError — unused in this repo
