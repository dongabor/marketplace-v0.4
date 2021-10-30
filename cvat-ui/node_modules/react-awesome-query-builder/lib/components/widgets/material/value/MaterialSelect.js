"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireDefault(require("react"));

var _Select = _interopRequireDefault(require("@material-ui/core/Select"));

var _MenuItem = _interopRequireDefault(require("@material-ui/core/MenuItem"));

var _stuff = require("../../../../utils/stuff");

var _FormControl = _interopRequireDefault(require("@material-ui/core/FormControl"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

var _default = function _default(_ref) {
  var listValues = _ref.listValues,
      value = _ref.value,
      setValue = _ref.setValue,
      allowCustomValues = _ref.allowCustomValues,
      readonly = _ref.readonly,
      placeholder = _ref.placeholder,
      customProps = _ref.customProps;

  var renderOptions = function renderOptions() {
    return (0, _stuff.mapListValues)(listValues, function (_ref2) {
      var title = _ref2.title,
          value = _ref2.value;
      return /*#__PURE__*/_react["default"].createElement(_MenuItem["default"], {
        key: value,
        value: value
      }, title);
    });
  };

  var onChange = function onChange(e) {
    if (e.target.value === undefined) return;
    setValue(e.target.value);
  };

  var renderValue = function renderValue(selectedValue) {
    if (!readonly && !selectedValue) return placeholder;
    return (0, _stuff.mapListValues)(listValues, function (_ref3) {
      var title = _ref3.title,
          value = _ref3.value;
      return value === selectedValue ? title : null;
    }).filter(function (v) {
      return v !== null;
    }).shift();
  };

  var hasValue = value != null;
  return /*#__PURE__*/_react["default"].createElement(_FormControl["default"], null, /*#__PURE__*/_react["default"].createElement(_Select["default"], _extends({
    autoWidth: true,
    displayEmpty: true,
    label: !readonly ? placeholder : "",
    onChange: onChange,
    value: hasValue ? value : "",
    disabled: readonly,
    readOnly: readonly,
    renderValue: renderValue
  }, customProps), renderOptions()));
};

exports["default"] = _default;