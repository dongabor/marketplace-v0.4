"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireWildcard(require("react"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _antd = require("antd");

var _moment = _interopRequireDefault(require("moment"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _getRequireWildcardCache() { if (typeof WeakMap !== "function") return null; var cache = new WeakMap(); _getRequireWildcardCache = function _getRequireWildcardCache() { return cache; }; return cache; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } if (obj === null || _typeof(obj) !== "object" && typeof obj !== "function") { return { "default": obj }; } var cache = _getRequireWildcardCache(); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj["default"] = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); if (superClass) _setPrototypeOf(subClass, superClass); }

function _setPrototypeOf(o, p) { _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return _setPrototypeOf(o, p); }

function _createSuper(Derived) { var hasNativeReflectConstruct = _isNativeReflectConstruct(); return function _createSuperInternal() { var Super = _getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = _getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return _possibleConstructorReturn(this, result); }; }

function _possibleConstructorReturn(self, call) { if (call && (_typeof(call) === "object" || typeof call === "function")) { return call; } return _assertThisInitialized(self); }

function _assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }

function _isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Date.prototype.toString.call(Reflect.construct(Date, [], function () {})); return true; } catch (e) { return false; } }

function _getPrototypeOf(o) { _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return _getPrototypeOf(o); }

var RangePicker = _antd.DatePicker.RangePicker;

var DateWidget = /*#__PURE__*/function (_PureComponent) {
  _inherits(DateWidget, _PureComponent);

  var _super = _createSuper(DateWidget);

  function DateWidget(props) {
    var _this;

    _classCallCheck(this, DateWidget);

    _this = _super.call(this, props);

    _this.isValidSingleValue = function (value) {
      var valueFormat = _this.props.valueFormat;
      var v = value ? (0, _moment["default"])(value, valueFormat) : null;
      return !v || v && v.isValid();
    };

    _this.isValidValue = function (value) {
      var isSpecialRange = _this.props.isSpecialRange;
      if (isSpecialRange) return value ? value.map(function (el) {
        return _this.isValidSingleValue(el);
      }).reduce(function (res, item) {
        return res && item;
      }, true) : true;else return _this.isValidSingleValue(value);
    };

    _this.getMomentSingleValue = function (value) {
      var valueFormat = _this.props.valueFormat;
      var v = value ? (0, _moment["default"])(value, valueFormat) : null;
      if (v && !v.isValid()) v = null;
      return v;
    };

    _this.getMomentValue = function (value) {
      var isSpecialRange = _this.props.isSpecialRange;
      if (isSpecialRange) return value ? value.map(function (el) {
        return _this.getMomentSingleValue(el);
      }) : [null, null];else return _this.getMomentSingleValue(value);
    };

    _this.formatSingleValue = function (value) {
      var valueFormat = _this.props.valueFormat;
      return value && value.isValid() ? value.format(valueFormat) : undefined;
    };

    _this.formatValue = function (value) {
      var isSpecialRange = _this.props.isSpecialRange;
      if (isSpecialRange) return value ? value.map(function (el) {
        return _this.formatSingleValue(el);
      }) : [undefined, undefined];else return _this.formatSingleValue(value);
    };

    _this.handleChange = function (value) {
      var setValue = _this.props.setValue;
      if (_this.isValidValue(value)) setValue(_this.formatValue(value));
    };

    var _value = props.value,
        _setValue = props.setValue;

    if (!_this.isValidValue(_value)) {
      _setValue(_this.formatValue(_this.getMomentValue(_value)));
    }

    return _this;
  }

  _createClass(DateWidget, [{
    key: "render",
    value: function render() {
      var _this$props = this.props,
          placeholder = _this$props.placeholder,
          placeholders = _this$props.placeholders,
          customProps = _this$props.customProps,
          value = _this$props.value,
          dateFormat = _this$props.dateFormat,
          config = _this$props.config,
          readonly = _this$props.readonly,
          isSpecialRange = _this$props.isSpecialRange;
      var renderSize = config.settings.renderSize;
      var dateValue = this.getMomentValue(value);

      if (isSpecialRange) {
        return /*#__PURE__*/_react["default"].createElement(RangePicker, _extends({
          disabled: readonly,
          key: "widget-date",
          placeholder: placeholders,
          size: renderSize,
          format: dateFormat,
          value: dateValue,
          onChange: this.handleChange
        }, customProps));
      } else {
        return /*#__PURE__*/_react["default"].createElement(_antd.DatePicker, _extends({
          disabled: readonly,
          key: "widget-date",
          placeholder: placeholder,
          size: renderSize,
          format: dateFormat,
          value: dateValue,
          onChange: this.handleChange
        }, customProps));
      }
    }
  }]);

  return DateWidget;
}(_react.PureComponent);

exports["default"] = DateWidget;
DateWidget.propTypes = {
  setValue: _propTypes["default"].func.isRequired,
  value: _propTypes["default"].oneOfType([_propTypes["default"].string, _propTypes["default"].arrayOf(_propTypes["default"].string)]),
  //in valueFormat
  field: _propTypes["default"].string.isRequired,
  config: _propTypes["default"].object.isRequired,
  placeholder: _propTypes["default"].string,
  placeholders: _propTypes["default"].arrayOf(_propTypes["default"].string),
  customProps: _propTypes["default"].object,
  readonly: _propTypes["default"].bool,
  // from fieldSettings:
  dateFormat: _propTypes["default"].string,
  valueFormat: _propTypes["default"].string
};
DateWidget.defaultProps = {
  dateFormat: "YYYY-MM-DD",
  valueFormat: "YYYY-MM-DD"
};