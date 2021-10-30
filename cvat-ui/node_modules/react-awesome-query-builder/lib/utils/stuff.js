"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.isJsonLogic = exports.removePrefixPath = exports.normalizeListValues = exports.flatizeTreeData = exports.defaultTreeDataMap = exports.mapListValues = exports.getValueInListValues = exports.getTitleInListValues = exports.getItemInListValues = exports.useOnPropsChanged = exports.escapeRegExp = exports.shallowEqual = exports.eqArrSet = exports.eqSet = exports.deepEqual = exports.immutableEqual = exports.BUILT_IN_PLACEMENTS = exports.truncateString = exports.calcTextWidth = exports.bindActionCreators = exports.defaultValue = exports.SELECT_WIDTH_OFFSET_RIGHT = void 0;

var _mapValues = _interopRequireDefault(require("lodash/mapValues"));

var _immutable = _interopRequireWildcard(require("immutable"));

var _react = _interopRequireDefault(require("react"));

function _getRequireWildcardCache() { if (typeof WeakMap !== "function") return null; var cache = new WeakMap(); _getRequireWildcardCache = function _getRequireWildcardCache() { return cache; }; return cache; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } if (obj === null || _typeof(obj) !== "object" && typeof obj !== "function") { return { "default": obj }; } var cache = _getRequireWildcardCache(); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj["default"] = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

function _createForOfIteratorHelper(o, allowArrayLike) { var it; if (typeof Symbol === "undefined" || o[Symbol.iterator] == null) { if (Array.isArray(o) || (it = _unsupportedIterableToArray(o)) || allowArrayLike && o && typeof o.length === "number") { if (it) o = it; var i = 0; var F = function F() {}; return { s: F, n: function n() { if (i >= o.length) return { done: true }; return { done: false, value: o[i++] }; }, e: function e(_e) { throw _e; }, f: F }; } throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); } var normalCompletion = true, didErr = false, err; return { s: function s() { it = o[Symbol.iterator](); }, n: function n() { var step = it.next(); normalCompletion = step.done; return step; }, e: function e(_e2) { didErr = true; err = _e2; }, f: function f() { try { if (!normalCompletion && it["return"] != null) it["return"](); } finally { if (didErr) throw err; } } }; }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

var SELECT_WIDTH_OFFSET_RIGHT = 48;
exports.SELECT_WIDTH_OFFSET_RIGHT = SELECT_WIDTH_OFFSET_RIGHT;
var DEFAULT_FONT_SIZE = "14px";
var DEFAULT_FONT_FAMILY = "'Helvetica Neue', Helvetica, Arial, sans-serif"; // RegExp.quote = function (str) {
//     return str.replace(/([.?*+^$[\]\\(){}|-])/g, "\\$1");
// };

var defaultValue = function defaultValue(value, _default) {
  return typeof value === "undefined" ? _default : value;
};

exports.defaultValue = defaultValue;

var bindActionCreators = function bindActionCreators(actionCreators, config, dispatch) {
  return (0, _mapValues["default"])(actionCreators, function (actionCreator) {
    return function () {
      for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
        args[_key] = arguments[_key];
      }

      return dispatch(actionCreator.apply(void 0, [config].concat(args)));
    };
  });
};

exports.bindActionCreators = bindActionCreators;

var calcTextWidth = function calcTextWidth(str) {
  var fontFamily = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : DEFAULT_FONT_FAMILY;
  var fontSize = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : DEFAULT_FONT_SIZE;
  var div = document.createElement("div");
  div.innerHTML = str;
  var css = {
    "position": "absolute",
    "float": "left",
    "white-space": "nowrap",
    "visibility": "hidden",
    "font-size": fontSize,
    "font-family": fontFamily
  };

  for (var k in css) {
    div.style[k] = css[k];
  }

  div = document.body.appendChild(div);
  var w = div.offsetWidth;
  document.body.removeChild(div);
  return w;
};

exports.calcTextWidth = calcTextWidth;

var truncateString = function truncateString(str, n, useWordBoundary) {
  if (!n || str.length <= n) {
    return str;
  }

  var subString = str.substr(0, n - 1);
  return (useWordBoundary ? subString.substr(0, subString.lastIndexOf(" ")) : subString) + "...";
};

exports.truncateString = truncateString;
var BUILT_IN_PLACEMENTS = {
  bottomLeft: {
    points: ["tl", "bl"],
    offset: [0, 4],
    overflow: {
      adjustX: 0,
      adjustY: 1
    }
  },
  bottomRight: {
    points: ["tr", "br"],
    offset: [0, 4],
    overflow: {
      adjustX: 1,
      adjustY: 1
    }
  },
  topLeft: {
    points: ["bl", "tl"],
    offset: [0, -4],
    overflow: {
      adjustX: 0,
      adjustY: 1
    }
  },
  topRight: {
    points: ["br", "tr"],
    offset: [0, -4],
    overflow: {
      adjustX: 1,
      adjustY: 1
    }
  }
};
exports.BUILT_IN_PLACEMENTS = BUILT_IN_PLACEMENTS;

var immutableEqual = function immutableEqual(v1, v2) {
  if (v1 === v2) {
    return true;
  } else {
    return v1.equals(v2);
  }
};

exports.immutableEqual = immutableEqual;

var deepEqual = function deepEqual(v1, v2) {
  if (v1 === v2) {
    return true;
  } else if (_immutable.Map.isMap(v1)) {
    return v1.equals(v2);
  } else {
    return JSON.stringify(v1) == JSON.stringify(v2);
  }
}; //Do sets have same values?


exports.deepEqual = deepEqual;

var eqSet = function eqSet(as, bs) {
  if (as.size !== bs.size) return false;

  var _iterator = _createForOfIteratorHelper(as),
      _step;

  try {
    for (_iterator.s(); !(_step = _iterator.n()).done;) {
      var a = _step.value;
      if (!bs.has(a)) return false;
    }
  } catch (err) {
    _iterator.e(err);
  } finally {
    _iterator.f();
  }

  return true;
}; //Do arrays have same values?


exports.eqSet = eqSet;

var eqArrSet = function eqArrSet(arr1, arr2) {
  return eqSet(new Set(arr1), new Set(arr2));
};

exports.eqArrSet = eqArrSet;

var shallowEqual = function shallowEqual(a, b) {
  var deep = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;

  if (a === b) {
    return true;
  } else if (Array.isArray(a)) return shallowEqualArrays(a, b, deep);else if (_immutable.Map.isMap(a)) return a.equals(b);else if (_typeof(a) == "object") return shallowEqualObjects(a, b, deep);else return a === b;
};

exports.shallowEqual = shallowEqual;

function shallowEqualArrays(arrA, arrB) {
  var deep = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;

  if (arrA === arrB) {
    return true;
  }

  if (!arrA || !arrB) {
    return false;
  }

  var len = arrA.length;

  if (arrB.length !== len) {
    return false;
  }

  for (var i = 0; i < len; i++) {
    var isEqual = deep ? shallowEqual(arrA[i], arrB[i]) : arrA[i] === arrB[i];

    if (!isEqual) {
      return false;
    }
  }

  return true;
}

function shallowEqualObjects(objA, objB) {
  var deep = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;

  if (objA === objB) {
    return true;
  }

  if (!objA || !objB) {
    return false;
  }

  var aKeys = Object.keys(objA);
  var bKeys = Object.keys(objB);
  var len = aKeys.length;

  if (bKeys.length !== len) {
    return false;
  }

  for (var i = 0; i < len; i++) {
    var key = aKeys[i];
    var isEqual = deep ? shallowEqual(objA[key], objB[key], deep) : objA[key] === objB[key];

    if (!isEqual) {
      return false;
    }
  }

  return true;
}

var escapeRegExp = function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&"); // $& means the whole matched string
};

exports.escapeRegExp = escapeRegExp;

var canUseUnsafe = function canUseUnsafe() {
  var v = _react["default"].version.split(".").map(parseInt.bind(null, 10));

  return v[0] >= 16 && v[1] >= 3;
};

var useOnPropsChanged = function useOnPropsChanged(obj) {
  if (canUseUnsafe) {
    obj.UNSAFE_componentWillReceiveProps = function (nextProps) {
      obj.onPropsChanged(nextProps);
    };
  } else {
    obj.componentWillReceiveProps = function (nextProps) {
      obj.onPropsChanged(nextProps);
    };
  }
};

exports.useOnPropsChanged = useOnPropsChanged;

var isObject = function isObject(v) {
  return _typeof(v) == "object" && v !== null;
};

var listValue = function listValue(v, title) {
  return isObject(v) ? v : {
    value: v,
    title: title !== undefined ? title : v
  };
}; // convert {<value>: <title>, ..} or [value, ..] to normal [{value, title}, ..]


var listValuesToArray = function listValuesToArray(listValuesObj) {
  if (!isObject(listValuesObj)) return listValuesObj;
  if (Array.isArray(listValuesObj)) return listValuesObj.map(function (v) {
    return listValue(v);
  });
  var listValuesArr = [];

  for (var v in listValuesObj) {
    var title = listValuesObj[v];
    listValuesArr.push(listValue(v, title));
  }

  return listValuesArr;
}; // listValues can be {<value>: <title>, ..} or [{value, title}, ..] or [value, ..]


var getItemInListValues = function getItemInListValues(listValues, value) {
  if (Array.isArray(listValues)) {
    var values = listValues.map(function (v) {
      return listValue(v);
    });
    return values.find(function (v) {
      return v.value === value;
    }) || values.find(function (v) {
      return "".concat(v.value) === value;
    });
  } else {
    return listValues[value] !== undefined ? listValue(value, listValues[value]) : undefined;
  }
};

exports.getItemInListValues = getItemInListValues;

var getTitleInListValues = function getTitleInListValues(listValues, value) {
  var it = getItemInListValues(listValues, value);
  return it !== undefined ? it.title : undefined;
};

exports.getTitleInListValues = getTitleInListValues;

var getValueInListValues = function getValueInListValues(listValues, value) {
  var it = getItemInListValues(listValues, value);
  return it !== undefined ? it.value : undefined;
};

exports.getValueInListValues = getValueInListValues;

var mapListValues = function mapListValues(listValues, fun) {
  var ret = [];

  if (Array.isArray(listValues)) {
    var _iterator2 = _createForOfIteratorHelper(listValues),
        _step2;

    try {
      for (_iterator2.s(); !(_step2 = _iterator2.n()).done;) {
        var v = _step2.value;
        ret.push(fun(listValue(v)));
      }
    } catch (err) {
      _iterator2.e(err);
    } finally {
      _iterator2.f();
    }
  } else {
    for (var value in listValues) {
      ret.push(fun(listValue(value, listValues[value])));
    }
  }

  return ret;
};

exports.mapListValues = mapListValues;
var defaultTreeDataMap = {
  id: "value",
  pId: "parent",
  rootPId: undefined
}; // converts from treeData to treeDataSimpleMode format (https://ant.design/components/tree-select/)
// ! modifies value of `treeData`

exports.defaultTreeDataMap = defaultTreeDataMap;

var flatizeTreeData = function flatizeTreeData(treeData) {
  var tdm = defaultTreeDataMap;
  var rind;
  var len;

  var _flatize = function _flatize(node, root, lev) {
    if (node.children) {
      if (lev == 1) node[tdm.pId] = tdm.rootPId; //optional?

      var childrenCount = node.children.length;

      var _iterator3 = _createForOfIteratorHelper(node.children),
          _step3;

      try {
        for (_iterator3.s(); !(_step3 = _iterator3.n()).done;) {
          var c = _step3.value;
          c[tdm.pId] = node[tdm.id];
          rind++;
          root.splice(rind, 0, c); //instead of just push

          len++;

          _flatize(c, root, lev + 1);
        }
      } catch (err) {
        _iterator3.e(err);
      } finally {
        _iterator3.f();
      }

      delete node.children;

      if (childrenCount == 0) {
        root.splice(rind, 1);
        rind--;
        len--;
      }
    }
  };

  if (Array.isArray(treeData)) {
    len = treeData.length;

    for (rind = 0; rind < len; rind++) {
      var c = treeData[rind];
      if (!isObject(c)) continue;
      if (c[tdm.pId] !== undefined && c[tdm.pId] != tdm.rootPId) continue; //not lev 1

      _flatize(c, treeData, 1);
    }
  }

  return treeData;
};

exports.flatizeTreeData = flatizeTreeData;

var getPathInListValues = function getPathInListValues(listValues, value) {
  var tdm = defaultTreeDataMap;
  var it = getItemInListValues(listValues, value);
  var parentId = it ? it[tdm.pId] : undefined;
  var parent = parentId ? listValues.find(function (v) {
    return v[tdm.id] === parentId;
  }) : undefined;
  return parent ? [parent.value].concat(_toConsumableArray(getPathInListValues(listValues, parent.value))) : [];
};

var getChildrenInListValues = function getChildrenInListValues(listValues, value) {
  var tdm = defaultTreeDataMap;
  var it = getItemInListValues(listValues, value);
  return it ? listValues.filter(function (v) {
    return v[tdm.pId] === it[tdm.id];
  }).map(function (v) {
    return v.value;
  }) : [];
}; // ! modifies value of `treeData`


var extendTreeData = function extendTreeData(treeData, fieldSettings, isMulti) {
  var _iterator4 = _createForOfIteratorHelper(treeData),
      _step4;

  try {
    for (_iterator4.s(); !(_step4 = _iterator4.n()).done;) {
      var node = _step4.value;
      node.path = getPathInListValues(treeData, node.value);

      if (fieldSettings.treeSelectOnlyLeafs != false) {
        var childrenValues = getChildrenInListValues(treeData, node.value);

        if (!isMulti) {
          node.selectable = childrenValues.length == 0;
        }
      }
    }
  } catch (err) {
    _iterator4.e(err);
  } finally {
    _iterator4.f();
  }

  return treeData;
};

var normalizeListValues = function normalizeListValues(listValues, type, fieldSettings) {
  var isTree = ["treeselect", "treemultiselect"].includes(type);
  var isMulti = ["multiselect", "treemultiselect"].includes(type);

  if (isTree) {
    listValues = listValuesToArray(listValues);
    listValues = flatizeTreeData(listValues);
    listValues = extendTreeData(listValues, fieldSettings, isMulti);
  }

  return listValues;
};

exports.normalizeListValues = normalizeListValues;

var removePrefixPath = function removePrefixPath(selectedPath, parentPath) {
  if (!selectedPath) return selectedPath;
  var isPrefix = true;

  for (var i = 0; i < parentPath.length; i++) {
    var part = parentPath[i];

    if (selectedPath[i] !== undefined && part == selectedPath[i]) {//ok
    } else {
      isPrefix = false;
      break;
    }
  }

  return isPrefix ? selectedPath.slice(parentPath.length) : selectedPath;
};

exports.removePrefixPath = removePrefixPath;

var isJsonLogic = function isJsonLogic(logic) {
  return _typeof(logic) === "object" // An object
  && logic !== null // but not null
  && !Array.isArray(logic) // and not an array
  && Object.keys(logic).length === 1 // with exactly one key
  ;
};

exports.isJsonLogic = isJsonLogic;