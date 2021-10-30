"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireDefault(require("react"));

var _Slider = _interopRequireDefault(require("@material-ui/core/Slider"));

var _TextField = _interopRequireDefault(require("@material-ui/core/TextField"));

var _FormControl = _interopRequireDefault(require("@material-ui/core/FormControl"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

function _objectWithoutProperties(source, excluded) { if (source == null) return {}; var target = _objectWithoutPropertiesLoose(source, excluded); var key, i; if (Object.getOwnPropertySymbols) { var sourceSymbolKeys = Object.getOwnPropertySymbols(source); for (i = 0; i < sourceSymbolKeys.length; i++) { key = sourceSymbolKeys[i]; if (excluded.indexOf(key) >= 0) continue; if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue; target[key] = source[key]; } } return target; }

function _objectWithoutPropertiesLoose(source, excluded) { if (source == null) return {}; var target = {}; var sourceKeys = Object.keys(source); var key, i; for (i = 0; i < sourceKeys.length; i++) { key = sourceKeys[i]; if (excluded.indexOf(key) >= 0) continue; target[key] = source[key]; } return target; }

var _default = function _default(props) {
  var placeholder = props.placeholder,
      customProps = props.customProps,
      value = props.value,
      setValue = props.setValue,
      min = props.min,
      max = props.max,
      step = props.step,
      marks = props.marks,
      readonly = props.readonly;

  var handleSliderChange = function handleSliderChange(_e, newValue) {
    setValue(newValue);
  };

  var handleInputChange = function handleInputChange(e) {
    var val = e.target.value;
    if (val === "" || val === null) val = undefined;else val = Number(val);
    setValue(val);
  };

  var handleInputBlur = function handleInputBlur() {
    // TIP: Fix if typed value out of range in input
    if (value < min) {
      setValue(min);
    } else if (value > max) {
      setValue(max);
    }
  };

  var _ref = customProps || {},
      width = _ref.width,
      rest = _objectWithoutProperties(_ref, ["width"]);

  var customInputProps = rest.input || {};
  var customSliderProps = rest.slider || rest; // TIP: Can't pass undefined to MUI, cause it means uncontrolled component use.
  //      For empty value input needs "", slider needs null or 0

  var inputValue = typeof value === "number" ? value : "";
  var sliderValue = typeof value === "number" ? value : null; // marks example: { 0: "0%", 100: React.createElement('strong', null, "100%") }

  var muiMarks = marks ? Object.keys(marks).map(function (v) {
    return {
      value: v,
      label: marks[v]
    };
  }) : false;

  var InputCmp = /*#__PURE__*/_react["default"].createElement(_TextField["default"], _extends({
    type: "number",
    value: inputValue,
    placeholder: placeholder,
    InputProps: {
      readOnly: readonly
    },
    inputProps: {
      min: min,
      max: max,
      step: step
    },
    disabled: readonly,
    onChange: handleInputChange,
    onBlur: handleInputBlur
  }, customInputProps));

  var SliderCmp = /*#__PURE__*/_react["default"].createElement(_Slider["default"], _extends({
    value: sliderValue,
    onChange: handleSliderChange,
    disabled: readonly,
    min: min,
    max: max,
    step: step,
    marks: muiMarks,
    valueLabelDisplay: "auto"
  }, customSliderProps));

  var stylesWrapper = {
    display: "inline-flex",
    alignItems: "center"
  };
  var stylesInputWrapper = {
    marginLeft: "5px"
  };
  var stylesSliderWrapper = {
    marginLeft: "5px",
    paddingLeft: "12px",
    marginBottom: muiMarks && "-16px",
    width: width || "300px"
  };
  return /*#__PURE__*/_react["default"].createElement(_FormControl["default"], null, /*#__PURE__*/_react["default"].createElement("div", {
    style: stylesWrapper
  }, /*#__PURE__*/_react["default"].createElement("div", {
    style: stylesInputWrapper
  }, InputCmp), /*#__PURE__*/_react["default"].createElement("div", {
    style: stylesSliderWrapper
  }, SliderCmp)));
};

exports["default"] = _default;