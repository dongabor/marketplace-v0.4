"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.getNewValueForFieldOp = exports.validateValue = exports.validateTree = void 0;

var _configUtils = require("./configUtils");

var _stuff = require("../utils/stuff");

var _defaultUtils = require("../utils/defaultUtils");

var _omit = _interopRequireDefault(require("lodash/omit"));

var _immutable = _interopRequireDefault(require("immutable"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArrayLimit(arr, i) { if (typeof Symbol === "undefined" || !(Symbol.iterator in Object(arr))) return; var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

var typeOf = function typeOf(v) {
  if (_typeof(v) == "object" && v !== null && Array.isArray(v)) return "array";else return _typeof(v);
};

var isTypeOf = function isTypeOf(v, type) {
  if (typeOf(v) == type) return true;
  if (type == "number" && !isNaN(v)) return true; //can be casted

  return false;
};

var validateTree = function validateTree(tree, _oldTree, config, oldConfig) {
  var removeEmptyGroups = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : false;
  var removeInvalidRules = arguments.length > 5 && arguments[5] !== undefined ? arguments[5] : false;
  var c = {
    config: config,
    oldConfig: oldConfig,
    removeEmptyGroups: removeEmptyGroups,
    removeInvalidRules: removeInvalidRules
  };
  return validateItem(tree, [], null, {}, c);
};

exports.validateTree = validateTree;

function validateItem(item, path, itemId, meta, c) {
  var type = item.get("type");
  var children = item.get("children1");

  if ((type === "group" || type === "rule_group") && children && children.size) {
    return validateGroup(item, path, itemId, meta, c);
  } else if (type === "rule") {
    return validateRule(item, path, itemId, meta, c);
  } else {
    return item;
  }
}

function validateGroup(item, path, itemId, meta, c) {
  var removeEmptyGroups = c.removeEmptyGroups;
  var id = item.get("id");
  var children = item.get("children1");
  var oldChildren = children;

  if (!id && itemId) {
    id = itemId;
    item = item.set("id", id);
    meta.sanitized = true;
  } //validate children


  var submeta = {};
  children = children.map(function (currentChild, childId) {
    return validateItem(currentChild, path.concat(id), childId, submeta, c);
  });
  if (removeEmptyGroups) children = children.filter(function (currentChild) {
    return currentChild != undefined;
  });
  var sanitized = submeta.sanitized || oldChildren.size != children.size;

  if (!children.size && removeEmptyGroups && path.length) {
    sanitized = true;
    item = undefined;
  }

  if (sanitized) meta.sanitized = true;
  if (sanitized && item) item = item.set("children1", children);
  return item;
}

function validateRule(item, path, itemId, meta, c) {
  var removeInvalidRules = c.removeInvalidRules,
      config = c.config,
      oldConfig = c.oldConfig;
  var showErrorMessage = config.settings.showErrorMessage;
  var id = item.get("id");
  var properties = item.get("properties");
  var field = properties.get("field");
  var operator = properties.get("operator");
  var operatorOptions = properties.get("operatorOptions");
  var valueSrc = properties.get("valueSrc");
  var value = properties.get("value");
  var valueError = properties.get("valueError");
  var oldSerialized = {
    field: field,
    operator: operator,
    operatorOptions: operatorOptions ? operatorOptions.toJS() : {},
    valueSrc: valueSrc ? valueSrc.toJS() : null,
    value: value ? value.toJS() : null,
    valueError: valueError ? valueError.toJS() : null
  };

  var _wasValid = field && operator && value && !value.find(function (v, ind) {
    return v === undefined;
  });

  if (!id && itemId) {
    id = itemId;
    item = item.set("id", id);
    meta.sanitized = true;
  } //validate field


  var fieldDefinition = field ? (0, _configUtils.getFieldConfig)(field, config) : null;
  if (!fieldDefinition) field = null;

  if (field == null) {
    properties = ["operator", "operatorOptions", "valueSrc", "value"].reduce(function (map, key) {
      return map["delete"](key);
    }, properties);
    operator = null;
  } //validate operator


  operator = properties.get("operator");

  if (operator == "range_between" || operator == "range_not_between") {
    // fix obsolete operators
    operator = operator == "range_between" ? "between" : "not_between";
    properties = properties.set("operator", operator);
  }

  var operatorDefinition = operator ? (0, _configUtils.getOperatorConfig)(config, operator, field) : null;
  if (!operatorDefinition) operator = null;
  var availOps = field ? (0, _configUtils.getOperatorsForField)(config, field) : [];

  if (!availOps) {
    console.warn("Type of field ".concat(field, " is not supported"));
    operator = null;
  } else if (availOps.indexOf(operator) == -1) {
    operator = null;
  }

  if (operator == null) {
    properties = properties["delete"]("operatorOptions");
    properties = properties["delete"]("valueSrc");
    properties = properties["delete"]("value");
  } //validate operator options


  operatorOptions = properties.get("operatorOptions");

  var _operatorCardinality = operator ? (0, _stuff.defaultValue)(operatorDefinition.cardinality, 1) : null;

  if (!operator || operatorOptions && !operatorDefinition.options) {
    operatorOptions = null;
    properties = properties["delete"]("operatorOptions");
  } else if (operator && !operatorOptions && operatorDefinition.options) {
    operatorOptions = (0, _defaultUtils.defaultOperatorOptions)(config, operator, field);
    properties = properties.set("operatorOptions", operatorOptions);
  } //validate values


  valueSrc = properties.get("valueSrc");
  value = properties.get("value");

  var _getNewValueForFieldO = getNewValueForFieldOp(config, oldConfig, properties, field, operator, null, true),
      newValue = _getNewValueForFieldO.newValue,
      newValueSrc = _getNewValueForFieldO.newValueSrc,
      newValueError = _getNewValueForFieldO.newValueError;

  value = newValue;
  valueSrc = newValueSrc;
  valueError = newValueError;
  properties = properties.set("value", value);
  properties = properties.set("valueSrc", valueSrc);

  if (showErrorMessage) {
    properties = properties.set("valueError", valueError);
  }

  var newSerialized = {
    field: field,
    operator: operator,
    operatorOptions: operatorOptions ? operatorOptions.toJS() : {},
    valueSrc: valueSrc ? valueSrc.toJS() : null,
    value: value ? value.toJS() : null,
    valueError: valueError ? valueError.toJS() : null
  };
  var sanitized = !(0, _stuff.deepEqual)(oldSerialized, newSerialized);
  var isValid = field && operator && value && !value.find(function (v, _ind) {
    return v === undefined;
  });
  if (sanitized) meta.sanitized = true;
  if (sanitized && !isValid && removeInvalidRules) item = undefined;
  if (sanitized && item) item = item.set("properties", properties);
  return item;
}
/**
 * 
 * @param {bool} canFix true is useful for func values to remove bad args
 * @param {bool} isEndValue false if value is in process of editing by user
 * @param {bool} isRawValue false is used only internally from validateFuncValue
 * @return {array} [validError, fixedValue] - if validError === null and canFix == true, fixedValue can differ from value if was fixed
 */


var validateValue = function validateValue(config, leftField, field, operator, value, valueType, valueSrc) {
  var canFix = arguments.length > 7 && arguments[7] !== undefined ? arguments[7] : false;
  var isEndValue = arguments.length > 8 && arguments[8] !== undefined ? arguments[8] : false;
  var isRawValue = arguments.length > 9 && arguments[9] !== undefined ? arguments[9] : true;
  var validError = null;
  var fixedValue = value;

  if (value != null) {
    if (valueSrc == "field") {
      var _validateFieldValue = validateFieldValue(leftField, field, value, valueSrc, valueType, config, operator, isEndValue, canFix);

      var _validateFieldValue2 = _slicedToArray(_validateFieldValue, 2);

      validError = _validateFieldValue2[0];
      fixedValue = _validateFieldValue2[1];
    } else if (valueSrc == "func") {
      var _validateFuncValue = validateFuncValue(leftField, field, value, valueSrc, valueType, config, operator, isEndValue, canFix);

      var _validateFuncValue2 = _slicedToArray(_validateFuncValue, 2);

      validError = _validateFuncValue2[0];
      fixedValue = _validateFuncValue2[1];
    } else if (valueSrc == "value" || !valueSrc) {
      var _validateNormalValue = validateNormalValue(leftField, field, value, valueSrc, valueType, config, operator, isEndValue, canFix);

      var _validateNormalValue2 = _slicedToArray(_validateNormalValue, 2);

      validError = _validateNormalValue2[0];
      fixedValue = _validateNormalValue2[1];
    }

    if (!validError) {
      var fieldConfig = (0, _configUtils.getFieldConfig)(field, config);
      var w = (0, _configUtils.getWidgetForFieldOp)(config, field, operator, valueSrc);
      var fieldWidgetDefinition = (0, _omit["default"])((0, _configUtils.getFieldWidgetConfig)(config, field, operator, w, valueSrc), ["factory"]);
      var rightFieldDefinition = valueSrc == "field" ? (0, _configUtils.getFieldConfig)(value, config) : null;
      var fieldSettings = fieldWidgetDefinition; // widget definition merged with fieldSettings

      var fn = fieldWidgetDefinition.validateValue;

      if (typeof fn == "function") {
        var args = [fixedValue, fieldSettings];
        if (valueSrc == "field") args.push(rightFieldDefinition);
        var validResult = fn.apply(void 0, args);

        if (typeof validResult == "boolean") {
          if (validResult == false) validError = "Invalid value";
        } else {
          validError = validResult;
        }
      }
    }
  }

  if (isRawValue && validError) {
    console.warn("[RAQB validate]", "Field ".concat(field, ": ").concat(validError));
  }

  return [validError, validError ? value : fixedValue];
};
/**
* 
*/


exports.validateValue = validateValue;

var validateNormalValue = function validateNormalValue(leftField, field, value, valueSrc, valueType, config) {
  var operator = arguments.length > 6 && arguments[6] !== undefined ? arguments[6] : null;
  var isEndValue = arguments.length > 7 && arguments[7] !== undefined ? arguments[7] : false;
  var canFix = arguments.length > 8 && arguments[8] !== undefined ? arguments[8] : false;
  var fixedValue = value;
  var fieldConfig = (0, _configUtils.getFieldConfig)(field, config);
  var w = (0, _configUtils.getWidgetForFieldOp)(config, field, operator, valueSrc);
  var wConfig = config.widgets[w];
  var wType = wConfig.type;
  var jsType = wConfig.jsType;
  var fieldSettings = fieldConfig.fieldSettings;
  if (valueType != wType) return ["Value should have type ".concat(wType, ", but got value of type ").concat(valueType), value];

  if (jsType && !isTypeOf(value, jsType) && !fieldSettings.listValues) {
    //tip: can skip tye check for listValues
    return ["Value should have JS type ".concat(jsType, ", but got value of type ").concat(_typeof(value)), value];
  }

  if (fieldSettings) {
    if (fieldSettings.listValues && !fieldSettings.allowCustomValues) {
      if (value instanceof Array) {
        for (var i = 0; i < value.length; i++) {
          var vv = (0, _stuff.getItemInListValues)(fieldSettings.listValues, value[i]);

          if (vv == undefined) {
            return ["Value ".concat(value[i], " is not in list of values"), value];
          } else {
            value[i] = vv.value;
          }
        }
      } else {
        var _vv = (0, _stuff.getItemInListValues)(fieldSettings.listValues, value);

        if (_vv == undefined) {
          return ["Value ".concat(value, " is not in list of values"), value];
        } else {
          value = _vv.value;
        }
      }
    }

    if (fieldSettings.min != null && value < fieldSettings.min) {
      return ["Value ".concat(value, " < min ").concat(fieldSettings.min), value];
    }

    if (fieldSettings.max != null && value > fieldSettings.max) {
      return ["Value ".concat(value, " > max ").concat(fieldSettings.max), value];
    }
  }

  return [null, value];
};
/**
* 
*/


var validateFieldValue = function validateFieldValue(leftField, field, value, _valueSrc, valueType, config) {
  var operator = arguments.length > 6 && arguments[6] !== undefined ? arguments[6] : null;
  var isEndValue = arguments.length > 7 && arguments[7] !== undefined ? arguments[7] : false;
  var canFix = arguments.length > 8 && arguments[8] !== undefined ? arguments[8] : false;
  var fieldSeparator = config.settings.fieldSeparator;
  var leftFieldStr = Array.isArray(leftField) ? leftField.join(fieldSeparator) : leftField;
  var rightFieldStr = Array.isArray(value) ? value.join(fieldSeparator) : value;
  var rightFieldDefinition = (0, _configUtils.getFieldConfig)(value, config);
  if (!rightFieldDefinition) return ["Unknown field ".concat(value), value];
  if (rightFieldStr == leftFieldStr) return ["Can't compare field ".concat(leftField, " with itself"), value];
  if (valueType && valueType != rightFieldDefinition.type) return ["Field ".concat(value, " is of type ").concat(rightFieldDefinition.type, ", but expected ").concat(valueType), value];
  return [null, value];
};
/**
* 
*/


var validateFuncValue = function validateFuncValue(leftField, field, value, _valueSrc, valueType, config) {
  var operator = arguments.length > 6 && arguments[6] !== undefined ? arguments[6] : null;
  var isEndValue = arguments.length > 7 && arguments[7] !== undefined ? arguments[7] : false;
  var canFix = arguments.length > 8 && arguments[8] !== undefined ? arguments[8] : false;
  var fixedValue = value;

  if (value) {
    var funcKey = value.get("func");

    if (funcKey) {
      var funcConfig = (0, _configUtils.getFuncConfig)(funcKey, config);

      if (funcConfig) {
        if (valueType && funcConfig.returnType != valueType) return ["Function ".concat(funcKey, " should return value of type ").concat(funcConfig.returnType, ", but got ").concat(valueType), value];

        for (var argKey in funcConfig.args) {
          var argConfig = funcConfig.args[argKey];
          var args = fixedValue.get("args");
          var argVal = args ? args.get(argKey) : undefined;
          var fieldDef = (0, _configUtils.getFieldConfig)(argConfig, config);
          var argValue = argVal ? argVal.get("value") : undefined;
          var argValueSrc = argVal ? argVal.get("valueSrc") : undefined;

          if (argValue !== undefined) {
            var _validateValue = validateValue(config, leftField, fieldDef, operator, argValue, argConfig.type, argValueSrc, canFix, isEndValue, false),
                _validateValue2 = _slicedToArray(_validateValue, 2),
                argValidError = _validateValue2[0],
                fixedArgVal = _validateValue2[1];

            if (argValidError !== null) {
              if (canFix) {
                fixedValue = fixedValue.deleteIn(["args", argKey]);

                if (argConfig.defaultValue !== undefined) {
                  fixedValue = fixedValue.setIn(["args", argKey, "value"], argConfig.defaultValue);
                  fixedValue = fixedValue.setIn(["args", argKey, "valueSrc"], "value");
                }
              } else {
                return ["Invalid value of arg ".concat(argKey, " for func ").concat(funcKey, ": ").concat(argValidError), value];
              }
            } else if (fixedArgVal !== argValue) {
              fixedValue = fixedValue.setIn(["args", argKey, "value"], fixedArgVal);
            }
          } else if (isEndValue && argConfig.defaultValue === undefined && !canFix) {
            return ["Value of arg ".concat(argKey, " for func ").concat(funcKey, " is required"), value];
          }
        }
      } else return ["Unknown function ".concat(funcKey), value];
    } // else it's not function value

  } // empty value


  return [null, fixedValue];
};
/**
 * @param {object} config
 * @param {object} oldConfig
 * @param {Immutable.Map} current
 * @param {string} newField
 * @param {string} newOperator
 * @param {string} changedField
 * @return {object} - {canReuseValue, newValue, newValueSrc, newValueType, newValueError}
 */


var getNewValueForFieldOp = function getNewValueForFieldOp(config) {
  var oldConfig = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : null;
  var current = arguments.length > 2 ? arguments[2] : undefined;
  var newField = arguments.length > 3 ? arguments[3] : undefined;
  var newOperator = arguments.length > 4 ? arguments[4] : undefined;
  var changedField = arguments.length > 5 && arguments[5] !== undefined ? arguments[5] : null;
  var canFix = arguments.length > 6 && arguments[6] !== undefined ? arguments[6] : true;
  if (!oldConfig) oldConfig = config;
  var currentField = current.get("field");
  var currentOperator = current.get("operator");
  var currentValue = current.get("value");
  var currentValueSrc = current.get("valueSrc", new _immutable["default"].List());
  var currentValueType = current.get("valueType", new _immutable["default"].List()); //const isValidatingTree = (changedField === null);

  var _config$settings = config.settings,
      convertableWidgets = _config$settings.convertableWidgets,
      clearValueOnChangeField = _config$settings.clearValueOnChangeField,
      clearValueOnChangeOp = _config$settings.clearValueOnChangeOp,
      showErrorMessage = _config$settings.showErrorMessage; //const currentOperatorConfig = getOperatorConfig(oldConfig, currentOperator, currentField);

  var newOperatorConfig = (0, _configUtils.getOperatorConfig)(config, newOperator, newField); //const currentOperatorCardinality = currentOperator ? defaultValue(currentOperatorConfig.cardinality, 1) : null;

  var operatorCardinality = newOperator ? (0, _stuff.defaultValue)(newOperatorConfig.cardinality, 1) : null;
  var currentFieldConfig = (0, _configUtils.getFieldConfig)(currentField, oldConfig);
  var newFieldConfig = (0, _configUtils.getFieldConfig)(newField, config);
  var canReuseValue = currentField && currentOperator && newOperator && (!changedField || changedField == "field" && !clearValueOnChangeField || changedField == "operator" && !clearValueOnChangeOp) && currentFieldConfig && newFieldConfig && currentFieldConfig.type == newFieldConfig.type; // compare old & new widgets

  for (var i = 0; i < operatorCardinality; i++) {
    var vs = currentValueSrc.get(i) || null;
    var currentWidget = (0, _configUtils.getWidgetForFieldOp)(oldConfig, currentField, currentOperator, vs);
    var newWidget = (0, _configUtils.getWidgetForFieldOp)(config, newField, newOperator, vs); // need to also check value widgets if we changed operator and current value source was 'field'
    // cause for select type op '=' requires single value and op 'in' requires array value

    var currentValueWidget = vs == "value" ? currentWidget : (0, _configUtils.getWidgetForFieldOp)(oldConfig, currentField, currentOperator, "value");
    var newValueWidget = vs == "value" ? newWidget : (0, _configUtils.getWidgetForFieldOp)(config, newField, newOperator, "value");
    var canReuseWidget = newValueWidget == currentValueWidget || (convertableWidgets[currentValueWidget] || []).includes(newValueWidget);
    if (!canReuseWidget) canReuseValue = false;
  }

  if (currentOperator != newOperator && [currentOperator, newOperator].includes("proximity")) canReuseValue = false;
  var firstWidgetConfig = (0, _configUtils.getFieldWidgetConfig)(config, newField, newOperator, null, currentValueSrc.first());
  var valueSources = (0, _configUtils.getValueSourcesForFieldOp)(config, newField, newOperator);
  var valueFixes = {};
  var valueErrors = Array.from({
    length: operatorCardinality
  }, function () {
    return null;
  });

  if (canReuseValue) {
    var _loop = function _loop(_i2) {
      var v = currentValue.get(_i2);
      var vType = currentValueType.get(_i2) || null;
      var vSrc = currentValueSrc.get(_i2) || null;
      var isValidSrc = valueSources.find(function (v) {
        return v == vSrc;
      }) != null;
      if (!isValidSrc && _i2 > 0 && vSrc == null) isValidSrc = true; // make exception for range widgets (when changing op from '==' to 'between')

      var isEndValue = !canFix;

      var _validateValue3 = validateValue(config, newField, newField, newOperator, v, vType, vSrc, canFix, isEndValue),
          _validateValue4 = _slicedToArray(_validateValue3, 2),
          validateError = _validateValue4[0],
          fixedValue = _validateValue4[1];

      var isValid = !validateError;

      if (!isValid && showErrorMessage && changedField != "field") {
        // allow bad value
        // but not on field change - in that case just drop bad value that can't be reused
        // ? maybe we should also drop bad value on op change?
        valueErrors[_i2] = validateError;
      } else if (!isValidSrc || !isValid) {
        canReuseValue = false;
        return "break";
      } else if (canFix && fixedValue !== v) {
        valueFixes[_i2] = fixedValue;
      }
    };

    for (var _i2 = 0; _i2 < operatorCardinality; _i2++) {
      var _ret = _loop(_i2);

      if (_ret === "break") break;
    }
  }

  var newValue = null,
      newValueSrc = null,
      newValueType = null,
      newValueError = null;
  newValue = new _immutable["default"].List(Array.from({
    length: operatorCardinality
  }, function (_ignore, i) {
    var v = undefined;

    if (canReuseValue) {
      if (i < currentValue.size) {
        v = currentValue.get(i);

        if (valueFixes[i] !== undefined) {
          v = valueFixes[i];
        }
      }
    } else if (operatorCardinality == 1 && (firstWidgetConfig || newFieldConfig)) {
      if (newFieldConfig.defaultValue !== undefined) v = newFieldConfig.defaultValue;else if (newFieldConfig.fieldSettings && newFieldConfig.fieldSettings.defaultValue !== undefined) v = newFieldConfig.fieldSettings.defaultValue;else if (firstWidgetConfig.defaultValue !== undefined) v = firstWidgetConfig.defaultValue;
    }

    return v;
  }));
  newValueSrc = new _immutable["default"].List(Array.from({
    length: operatorCardinality
  }, function (_ignore, i) {
    var vs = null;

    if (canReuseValue) {
      if (i < currentValueSrc.size) vs = currentValueSrc.get(i);
    } else if (valueSources.length == 1) {
      vs = valueSources[0];
    } else if (valueSources.length > 1) {
      vs = valueSources[0];
    }

    return vs;
  }));

  if (showErrorMessage) {
    if (newOperatorConfig && newOperatorConfig.validateValues && newValueSrc.toJS().filter(function (vs) {
      return vs == "value" || vs == null;
    }).length == operatorCardinality) {
      // last element in `valueError` list is for range validation error
      var jsValues = firstWidgetConfig && firstWidgetConfig.toJS ? newValue.toJS().map(function (v) {
        return firstWidgetConfig.toJS(v, firstWidgetConfig);
      }) : newValue.toJS();
      var rangeValidateError = newOperatorConfig.validateValues(jsValues);

      if (showErrorMessage) {
        valueErrors.push(rangeValidateError);
      }
    }

    newValueError = new _immutable["default"].List(valueErrors);
  }

  newValueType = new _immutable["default"].List(Array.from({
    length: operatorCardinality
  }, function (_ignore, i) {
    var vt = null;

    if (canReuseValue) {
      if (i < currentValueType.size) vt = currentValueType.get(i);
    } else if (operatorCardinality == 1 && firstWidgetConfig && firstWidgetConfig.type !== undefined) {
      vt = firstWidgetConfig.type;
    }

    return vt;
  }));
  return {
    canReuseValue: canReuseValue,
    newValue: newValue,
    newValueSrc: newValueSrc,
    newValueType: newValueType,
    newValueError: newValueError
  };
};

exports.getNewValueForFieldOp = getNewValueForFieldOp;