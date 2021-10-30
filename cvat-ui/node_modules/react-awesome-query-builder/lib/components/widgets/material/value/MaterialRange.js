"use strict";

function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireWildcard(require("react"));

var _Slider = _interopRequireDefault(require("@material-ui/core/Slider"));

var _TextField = _interopRequireDefault(require("@material-ui/core/TextField"));

var _FormControl = _interopRequireDefault(require("@material-ui/core/FormControl"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _getRequireWildcardCache() { if (typeof WeakMap !== "function") return null; var cache = new WeakMap(); _getRequireWildcardCache = function _getRequireWildcardCache() { return cache; }; return cache; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } if (obj === null || _typeof(obj) !== "object" && typeof obj !== "function") { return { "default": obj }; } var cache = _getRequireWildcardCache(); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj["default"] = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

function _objectWithoutProperties(source, excluded) { if (source == null) return {}; var target = _objectWithoutPropertiesLoose(source, excluded); var key, i; if (Object.getOwnPropertySymbols) { var sourceSymbolKeys = Object.getOwnPropertySymbols(source); for (i = 0; i < sourceSymbolKeys.length; i++) { key = sourceSymbolKeys[i]; if (excluded.indexOf(key) >= 0) continue; if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue; target[key] = source[key]; } } return target; }

function _objectWithoutPropertiesLoose(source, excluded) { if (source == null) return {}; var target = {}; var sourceKeys = Object.keys(source); var key, i; for (i = 0; i < sourceKeys.length; i++) { key = sourceKeys[i]; if (excluded.indexOf(key) >= 0) continue; target[key] = source[key]; } return target; }

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArrayLimit(arr, i) { if (typeof Symbol === "undefined" || !(Symbol.iterator in Object(arr))) return; var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

var _default = function _default(props) {
  var placeholders = props.placeholders,
      customProps = props.customProps,
      value = props.value,
      setValue = props.setValue,
      min = props.min,
      max = props.max,
      step = props.step,
      marks = props.marks,
      readonly = props.readonly,
      textSeparators = props.textSeparators;
  (0, _react.useEffect)(function () {
    var _ref = props.value || [undefined, undefined],
        _ref2 = _slicedToArray(_ref, 2),
        valueFrom = _ref2[0],
        valueTo = _ref2[1];

    if (props.value && (valueFrom == undefined || valueTo == undefined)) {
      // happens if we changed op from '==' to 'between'
      // (I know, timeout is dirty hack..)
      setTimeout(function () {
        var oneValue = valueFrom || valueTo;
        var value = [oneValue, oneValue];
        setValue(value);
      }, 1);
    }
  }, []);

  var handleSliderChange = function handleSliderChange(_e, newValues) {
    setValue(newValues);
  };

  var handleInputChangeFrom = function handleInputChangeFrom(e) {
    // TIP: need to use props.value instead of value
    var valueFrom = e.target.value;
    if (valueFrom === "" || valueFrom == null) valueFrom = undefined;else valueFrom = Number(valueFrom);
    var value = props.value ? _toConsumableArray(props.value) : [undefined, undefined];
    value[0] = valueFrom;
    setValue(value);
  };

  var handleInputChangeTo = function handleInputChangeTo(e) {
    var valueTo = e.target.value;
    if (valueTo === "" || valueTo == null) valueTo = undefined;else valueTo = Number(valueTo);
    var value = props.value ? _toConsumableArray(props.value) : [undefined, undefined];
    value[1] = valueTo;
    setValue(value);
  };

  var handleInputBlur = function handleInputBlur() {
    // TIP: Fix if typed value out of range in inputs
    if (!value) return;

    if (value[0] < min) {
      setValue([min, value[1]]);
    } else if (value[1] > max) {
      setValue([value[0], max]);
    }
  };

  var _ref3 = customProps || {},
      width = _ref3.width,
      rest = _objectWithoutProperties(_ref3, ["width"]);

  var customInputProps = rest.input || {};
  var customSliderProps = rest.slider || rest; // marks example: { 0: "0%", 100: React.createElement('strong', null, "100%") }

  var muiMarks = marks ? Object.keys(marks).map(function (v) {
    return {
      value: v,
      label: marks[v]
    };
  }) : false; // TIP: Can't pass undefined to MUI, cause it means uncontrolled component use.
  //      For empty value input needs "", slider needs null or 0, but null will cause problems with range mode

  var sliderValue = value ? _toConsumableArray(value) : [undefined, undefined];

  var _sliderValue = _slicedToArray(sliderValue, 2),
      valueFrom = _sliderValue[0],
      valueTo = _sliderValue[1];

  if (valueFrom == undefined) {
    valueFrom = "";
    sliderValue[0] = 0;
  }

  if (valueTo == undefined) {
    valueTo = "";
    sliderValue[1] = 0;
  }

  var FromInputCmp = /*#__PURE__*/_react["default"].createElement(_TextField["default"], _extends({
    type: "number",
    value: valueFrom,
    placeholder: placeholders[0],
    InputProps: {
      readOnly: readonly
    },
    inputProps: {
      min: min,
      max: max,
      step: step
    },
    disabled: readonly,
    onChange: handleInputChangeFrom,
    onBlur: handleInputBlur
  }, customInputProps));

  var ToInputCmp = /*#__PURE__*/_react["default"].createElement(_TextField["default"], _extends({
    type: "number",
    value: valueTo,
    placeholder: placeholders[1],
    InputProps: {
      readOnly: readonly
    },
    inputProps: {
      min: min,
      max: max,
      step: step
    },
    disabled: readonly,
    onChange: handleInputChangeTo,
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
    display: "inline-flex"
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
  }, FromInputCmp), /*#__PURE__*/_react["default"].createElement("div", {
    className: "widget--sep"
  }, /*#__PURE__*/_react["default"].createElement("span", null, textSeparators[1])), /*#__PURE__*/_react["default"].createElement("div", {
    style: stylesInputWrapper
  }, ToInputCmp), /*#__PURE__*/_react["default"].createElement("div", {
    style: stylesSliderWrapper
  }, SliderCmp)));
};

exports["default"] = _default;