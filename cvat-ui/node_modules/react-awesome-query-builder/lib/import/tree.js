"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
Object.defineProperty(exports, "isJsonLogic", {
  enumerable: true,
  get: function get() {
    return _stuff.isJsonLogic;
  }
});
exports.isTree = exports.isImmutableTree = exports.isValidTree = exports.checkTree = exports.loadTree = exports.getTree = void 0;

var _immutable = _interopRequireWildcard(require("immutable"));

var _validation = require("../utils/validation");

var _configUtils = require("../utils/configUtils");

var _treeUtils = require("../utils/treeUtils");

var _stuff = require("../utils/stuff");

function _getRequireWildcardCache() { if (typeof WeakMap !== "function") return null; var cache = new WeakMap(); _getRequireWildcardCache = function _getRequireWildcardCache() { return cache; }; return cache; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } if (obj === null || _typeof(obj) !== "object" && typeof obj !== "function") { return { "default": obj }; } var cache = _getRequireWildcardCache(); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj["default"] = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

var getTree = function getTree(immutableTree) {
  var light = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : true;
  if (!immutableTree) return undefined;
  var tree = immutableTree;
  tree = tree.toJS();
  if (light) tree = (0, _treeUtils.getLightTree)(tree);
  return tree;
};

exports.getTree = getTree;

var loadTree = function loadTree(serTree) {
  if (isImmutableTree(serTree)) {
    return serTree;
  } else if (isTree(serTree)) {
    return jsTreeToImmutable(serTree);
  } else if (typeof serTree == "string" && serTree.startsWith('["~#iM"')) {
    //tip: old versions of RAQB were saving tree with `transit.toJSON()`
    // https://github.com/ukrbublik/react-awesome-query-builder/issues/69
    throw "You are trying to load query in obsolete serialization format (Immutable string) which is not supported in versions starting from 2.1.17";
  } else if (typeof serTree == "string") {
    return jsTreeToImmutable(JSON.parse(serTree));
  } else throw "Can't load tree!";
};

exports.loadTree = loadTree;

var checkTree = function checkTree(tree, config) {
  if (!tree) return undefined;
  var extendedConfig = (0, _configUtils.extendConfig)(config);
  return (0, _validation.validateTree)(tree, null, extendedConfig, extendedConfig, true, true);
};

exports.checkTree = checkTree;

var isValidTree = function isValidTree(tree) {
  return (0, _treeUtils.getTreeBadFields)(tree).length == 0;
};

exports.isValidTree = isValidTree;

var isImmutableTree = function isImmutableTree(tree) {
  return _immutable.Map.isMap(tree);
};

exports.isImmutableTree = isImmutableTree;

var isTree = function isTree(tree) {
  return _typeof(tree) == "object" && tree.type == "group";
};

exports.isTree = isTree;

function jsTreeToImmutable(tree) {
  return (0, _immutable.fromJS)(tree, function (key, value) {
    var outValue;

    if (key == "value" && value.get(0) && value.get(0).toJS !== undefined) {
      var valueJs = value.get(0).toJS();

      if (valueJs.func) {
        outValue = value.toOrderedMap();
      } else {
        // only for raw values keep JS representation
        outValue = _immutable["default"].List.of(valueJs);
      }
    } else outValue = _immutable["default"].Iterable.isIndexed(value) ? value.toList() : value.toOrderedMap();

    return outValue;
  });
}