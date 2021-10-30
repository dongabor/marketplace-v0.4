"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireDefault(require("react"));

var _IconButton = _interopRequireDefault(require("@material-ui/core/IconButton"));

var _ExpandMoreSharp = _interopRequireDefault(require("@material-ui/icons/ExpandMoreSharp"));

var _Popover = _interopRequireDefault(require("@material-ui/core/Popover"));

var _Radio = _interopRequireDefault(require("@material-ui/core/Radio"));

var _RadioGroup = _interopRequireDefault(require("@material-ui/core/RadioGroup"));

var _FormControlLabel = _interopRequireDefault(require("@material-ui/core/FormControlLabel"));

var _FormControl = _interopRequireDefault(require("@material-ui/core/FormControl"));

var _FormLabel = _interopRequireDefault(require("@material-ui/core/FormLabel"));

var _styles = require("@material-ui/core/styles");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArrayLimit(arr, i) { if (typeof Symbol === "undefined" || !(Symbol.iterator in Object(arr))) return; var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

var useStyles = (0, _styles.makeStyles)(function (theme) {
  return {
    paper: {
      padding: theme.spacing(1)
    }
  };
});

var _default = function _default(_ref) {
  var valueSources = _ref.valueSources,
      valueSrc = _ref.valueSrc,
      title = _ref.title,
      setValueSrc = _ref.setValueSrc,
      readonly = _ref.readonly;

  var _React$useState = _react["default"].useState(null),
      _React$useState2 = _slicedToArray(_React$useState, 2),
      anchorEl = _React$useState2[0],
      setAnchorEl = _React$useState2[1];

  var classes = useStyles();

  var handleOpen = function handleOpen(event) {
    setAnchorEl(event.currentTarget);
  };

  var handleClose = function handleClose() {
    setAnchorEl(null);
  };

  var toggleOpenClose = function toggleOpenClose(event) {
    anchorEl ? handleClose() : handleOpen(event);
  };

  var handleChange = function handleChange(e) {
    if (e.target.value === undefined) return;
    setValueSrc(e.target.value);
    handleClose();
  };

  var renderOptions = function renderOptions(valueSources) {
    return valueSources.map(function (_ref2) {
      var _ref3 = _slicedToArray(_ref2, 2),
          srcKey = _ref3[0],
          info = _ref3[1];

      return /*#__PURE__*/_react["default"].createElement(_FormControlLabel["default"], {
        key: srcKey,
        value: srcKey,
        checked: valueSrc == srcKey || !valueSrc && srcKey == "value",
        control: /*#__PURE__*/_react["default"].createElement(_Radio["default"], null),
        label: info.label
      });
    });
  };

  var open = Boolean(anchorEl);
  return /*#__PURE__*/_react["default"].createElement("div", null, /*#__PURE__*/_react["default"].createElement(_IconButton["default"], {
    size: "small",
    onClick: toggleOpenClose
  }, /*#__PURE__*/_react["default"].createElement(_ExpandMoreSharp["default"], null)), /*#__PURE__*/_react["default"].createElement(_Popover["default"], {
    open: open,
    anchorEl: anchorEl,
    anchorOrigin: {
      vertical: "bottom",
      horizontal: "left"
    },
    transformOrigin: {
      vertical: "top",
      horizontal: "left"
    },
    onClose: handleClose,
    classes: {
      paper: classes.paper
    },
    disablePortal: true
  }, /*#__PURE__*/_react["default"].createElement(_FormControl["default"], {
    component: "fieldset"
  }, /*#__PURE__*/_react["default"].createElement(_FormLabel["default"], {
    component: "legend"
  }, title), /*#__PURE__*/_react["default"].createElement(_RadioGroup["default"], {
    value: valueSrc || "value",
    onChange: handleChange
  }, renderOptions(valueSources)))));
};

exports["default"] = _default;