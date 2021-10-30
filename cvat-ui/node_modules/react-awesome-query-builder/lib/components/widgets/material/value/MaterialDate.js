"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireDefault(require("react"));

var _pickers = require("@material-ui/pickers");

var _FormControl = _interopRequireDefault(require("@material-ui/core/FormControl"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

var _default = function _default(props) {
  var value = props.value,
      setValue = props.setValue,
      readonly = props.readonly,
      customProps = props.customProps,
      dateFormat = props.dateFormat,
      valueFormat = props.valueFormat,
      placeholder = props.placeholder,
      useKeyboard = props.useKeyboard;

  var formatSingleValue = function formatSingleValue(value) {
    return value && value.isValid() ? value.format(valueFormat) : undefined;
  };

  var handleChange = function handleChange(value) {
    setValue(formatSingleValue(value));
  };

  var Picker = useKeyboard ? _pickers.KeyboardDatePicker : _pickers.DatePicker;
  return /*#__PURE__*/_react["default"].createElement(_FormControl["default"], null, /*#__PURE__*/_react["default"].createElement(Picker, _extends({
    readOnly: readonly,
    disabled: readonly,
    placeholder: !readonly ? placeholder : "",
    format: dateFormat,
    value: value || null,
    onChange: handleChange
  }, customProps)));
};

exports["default"] = _default;