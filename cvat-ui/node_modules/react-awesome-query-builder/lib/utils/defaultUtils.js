"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.defaultRoot = exports.getChild = exports.defaultGroupProperties = exports.defaultConjunction = exports.defaultGroupConjunction = exports.defaultRuleProperties = exports.defaultOperatorOptions = exports.defaultOperator = exports.defaultField = void 0;

var _immutable = _interopRequireDefault(require("immutable"));

var _uuid = _interopRequireDefault(require("./uuid"));

var _configUtils = require("./configUtils");

var _validation = require("../utils/validation");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { _defineProperty(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

var defaultField = function defaultField(config) {
  var canGetFirst = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : true;
  var parentRuleGroupPath = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
  return typeof config.settings.defaultField === "function" ? config.settings.defaultField(parentRuleGroupPath) : config.settings.defaultField || (canGetFirst ? (0, _configUtils.getFirstField)(config, parentRuleGroupPath) : null);
};

exports.defaultField = defaultField;

var defaultOperator = function defaultOperator(config, field) {
  var canGetFirst = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : true;
  var fieldConfig = (0, _configUtils.getFieldConfig)(field, config);
  var fieldOperators = fieldConfig && fieldConfig.operators || [];
  var fieldDefaultOperator = fieldConfig && fieldConfig.defaultOperator;
  if (!fieldOperators.includes(fieldDefaultOperator)) fieldDefaultOperator = null;
  if (!fieldDefaultOperator && canGetFirst) fieldDefaultOperator = (0, _configUtils.getFirstOperator)(config, field);
  var op = typeof config.settings.defaultOperator === "function" ? config.settings.defaultOperator(field, fieldConfig) : fieldDefaultOperator;
  return op;
}; //used for complex operators like proximity


exports.defaultOperator = defaultOperator;

var defaultOperatorOptions = function defaultOperatorOptions(config, operator, field) {
  var operatorConfig = operator ? (0, _configUtils.getOperatorConfig)(config, operator, field) : null;
  if (!operatorConfig) return null; //new Immutable.Map();

  return operatorConfig.options ? new _immutable["default"].Map(operatorConfig.options && operatorConfig.options.defaults || {}) : null;
};

exports.defaultOperatorOptions = defaultOperatorOptions;

var defaultRuleProperties = function defaultRuleProperties(config) {
  var parentRuleGroupPath = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : null;
  var field = null,
      operator = null;
  var _config$settings = config.settings,
      setDefaultFieldAndOp = _config$settings.setDefaultFieldAndOp,
      showErrorMessage = _config$settings.showErrorMessage;

  if (setDefaultFieldAndOp) {
    field = defaultField(config, true, parentRuleGroupPath);
    operator = defaultOperator(config, field);
  }

  var current = new _immutable["default"].Map({
    field: field,
    operator: operator,
    value: new _immutable["default"].List(),
    valueSrc: new _immutable["default"].List(),
    //used for complex operators like proximity
    operatorOptions: defaultOperatorOptions(config, operator, field)
  });

  if (showErrorMessage) {
    current = current.set("valueError", new _immutable["default"].List());
  }

  if (field && operator) {
    var _getNewValueForFieldO = (0, _validation.getNewValueForFieldOp)(config, config, current, field, operator, "operator", false),
        newValue = _getNewValueForFieldO.newValue,
        newValueSrc = _getNewValueForFieldO.newValueSrc,
        newValueType = _getNewValueForFieldO.newValueType,
        newValueError = _getNewValueForFieldO.newValueError;

    current = current.set("value", newValue).set("valueSrc", newValueSrc).set("valueType", newValueType);

    if (showErrorMessage) {
      current = current.set("valueError", newValueError);
    }
  }

  return current;
}; //------------


exports.defaultRuleProperties = defaultRuleProperties;

var defaultGroupConjunction = function defaultGroupConjunction(config) {
  return config.settings.defaultGroupConjunction || config.settings.defaultConjunction || Object.keys(config.conjunctions)[0];
};

exports.defaultGroupConjunction = defaultGroupConjunction;

var defaultConjunction = function defaultConjunction(config) {
  return config.settings.defaultConjunction || Object.keys(config.conjunctions)[0];
};

exports.defaultConjunction = defaultConjunction;

var defaultGroupProperties = function defaultGroupProperties(config) {
  return new _immutable["default"].Map({
    conjunction: defaultGroupConjunction(config)
  });
}; //------------


exports.defaultGroupProperties = defaultGroupProperties;

var getChild = function getChild(id, config) {
  return _defineProperty({}, id, new _immutable["default"].Map({
    type: "rule",
    id: id,
    properties: defaultRuleProperties(config)
  }));
};

exports.getChild = getChild;

var defaultRoot = function defaultRoot(config) {
  if (config.tree) {
    return new _immutable["default"].Map(config.tree);
  }

  return new _immutable["default"].Map({
    type: "group",
    id: (0, _uuid["default"])(),
    children1: new _immutable["default"].OrderedMap(_objectSpread({}, getChild((0, _uuid["default"])(), config))),
    properties: defaultGroupProperties(config)
  });
};

exports.defaultRoot = defaultRoot;