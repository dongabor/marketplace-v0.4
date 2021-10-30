"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _en_US = _interopRequireDefault(require("antd/lib/locale-provider/en_US"));

var _antd = _interopRequireDefault(require("../../components/widgets/antd"));

var _basic = _interopRequireDefault(require("../basic"));

var _stuff = require("../../utils/stuff");

var _sql = require("../../utils/sql");

var _react = _interopRequireDefault(require("react"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { _defineProperty(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

var FieldSelect = _antd["default"].FieldSelect,
    FieldDropdown = _antd["default"].FieldDropdown,
    FieldCascader = _antd["default"].FieldCascader,
    FieldTreeSelect = _antd["default"].FieldTreeSelect,
    Button = _antd["default"].Button,
    ButtonGroup = _antd["default"].ButtonGroup,
    Conjs = _antd["default"].Conjs,
    ValueSources = _antd["default"].ValueSources,
    Provider = _antd["default"].Provider,
    confirm = _antd["default"].confirm;
var TextWidget = _antd["default"].TextWidget,
    NumberWidget = _antd["default"].NumberWidget,
    SliderWidget = _antd["default"].SliderWidget,
    RangeWidget = _antd["default"].RangeWidget,
    SelectWidget = _antd["default"].SelectWidget,
    MultiSelectWidget = _antd["default"].MultiSelectWidget,
    TreeSelectWidget = _antd["default"].TreeSelectWidget,
    DateWidget = _antd["default"].DateWidget,
    BooleanWidget = _antd["default"].BooleanWidget,
    TimeWidget = _antd["default"].TimeWidget,
    DateTimeWidget = _antd["default"].DateTimeWidget;

var settings = _objectSpread(_objectSpread({}, _basic["default"].settings), {}, {
  renderField: function renderField(props) {
    return /*#__PURE__*/_react["default"].createElement(FieldSelect, props);
  },
  // renderField: (props) => <FieldDropdown {...props} />,
  // renderField: (props) => <FieldCascader {...props} />,
  // renderField: (props) => <FieldTreeSelect {...props} />,
  renderOperator: function renderOperator(props) {
    return /*#__PURE__*/_react["default"].createElement(FieldSelect, props);
  },
  // renderOperator: (props) => <FieldDropdown {...props} />,
  renderFunc: function renderFunc(props) {
    return /*#__PURE__*/_react["default"].createElement(FieldSelect, props);
  },
  renderConjs: function renderConjs(props) {
    return /*#__PURE__*/_react["default"].createElement(Conjs, props);
  },
  renderButton: function renderButton(props) {
    return /*#__PURE__*/_react["default"].createElement(Button, props);
  },
  renderButtonGroup: function renderButtonGroup(props) {
    return /*#__PURE__*/_react["default"].createElement(ButtonGroup, props);
  },
  renderValueSources: function renderValueSources(props) {
    return /*#__PURE__*/_react["default"].createElement(ValueSources, props);
  },
  renderProvider: function renderProvider(props) {
    return /*#__PURE__*/_react["default"].createElement(Provider, props);
  },
  renderConfirm: confirm,
  // localization
  locale: _objectSpread(_objectSpread({}, _basic["default"].settings.locale), {}, {
    antd: _en_US["default"]
  })
});

var widgets = _objectSpread(_objectSpread({}, _basic["default"].widgets), {}, {
  text: _objectSpread(_objectSpread({}, _basic["default"].widgets.text), {}, {
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(TextWidget, props);
    }
  }),
  number: _objectSpread(_objectSpread({}, _basic["default"].widgets.number), {}, {
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(NumberWidget, props);
    }
  }),
  multiselect: _objectSpread(_objectSpread({}, _basic["default"].widgets.multiselect), {}, {
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(MultiSelectWidget, props);
    }
  }),
  select: _objectSpread(_objectSpread({}, _basic["default"].widgets.select), {}, {
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(SelectWidget, props);
    }
  }),
  slider: _objectSpread(_objectSpread({}, _basic["default"].widgets.slider), {}, {
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(SliderWidget, props);
    }
  }),
  "boolean": _objectSpread(_objectSpread({}, _basic["default"].widgets["boolean"]), {}, {
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(BooleanWidget, props);
    }
  }),
  date: _objectSpread(_objectSpread({}, _basic["default"].widgets.date), {}, {
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(DateWidget, props);
    }
  }),
  time: _objectSpread(_objectSpread({}, _basic["default"].widgets.time), {}, {
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(TimeWidget, props);
    }
  }),
  datetime: _objectSpread(_objectSpread({}, _basic["default"].widgets.datetime), {}, {
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(DateTimeWidget, props);
    }
  }),
  rangeslider: {
    type: "number",
    jsType: "number",
    valueSrc: "value",
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(RangeWidget, props);
    },
    valueLabel: "Range",
    valuePlaceholder: "Select range",
    valueLabels: [{
      label: "Number from",
      placeholder: "Enter number from"
    }, {
      label: "Number to",
      placeholder: "Enter number to"
    }],
    formatValue: function formatValue(val, fieldDef, wgtDef, isForDisplay) {
      return isForDisplay ? val : JSON.stringify(val);
    },
    sqlFormatValue: function sqlFormatValue(val, fieldDef, wgtDef, op, opDef) {
      return _sql.SqlString.escape(val);
    },
    singleWidget: "slider",
    toJS: function toJS(val, fieldSettings) {
      return val;
    }
  },
  treeselect: {
    type: "treeselect",
    jsType: "string",
    valueSrc: "value",
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(TreeSelectWidget, props);
    },
    valueLabel: "Value",
    valuePlaceholder: "Select value",
    formatValue: function formatValue(val, fieldDef, wgtDef, isForDisplay) {
      var valLabel = (0, _stuff.getTitleInListValues)(fieldDef.fieldSettings.listValues, val);
      return isForDisplay ? '"' + valLabel + '"' : JSON.stringify(val);
    },
    sqlFormatValue: function sqlFormatValue(val, fieldDef, wgtDef, op, opDef) {
      return _sql.SqlString.escape(val);
    },
    toJS: function toJS(val, fieldSettings) {
      return val;
    }
  },
  treemultiselect: {
    type: "treemultiselect",
    jsType: "array",
    valueSrc: "value",
    factory: function factory(props) {
      return /*#__PURE__*/_react["default"].createElement(TreeSelectWidget, _extends({}, props, {
        treeMultiple: true
      }));
    },
    valueLabel: "Values",
    valuePlaceholder: "Select values",
    formatValue: function formatValue(vals, fieldDef, wgtDef, isForDisplay) {
      var valsLabels = vals.map(function (v) {
        return (0, _stuff.getTitleInListValues)(fieldDef.fieldSettings.listValues, v);
      });
      return isForDisplay ? valsLabels.map(function (v) {
        return '"' + v + '"';
      }) : vals.map(function (v) {
        return JSON.stringify(v);
      });
    },
    sqlFormatValue: function sqlFormatValue(vals, fieldDef, wgtDef, op, opDef) {
      return vals.map(function (v) {
        return _sql.SqlString.escape(v);
      });
    },
    toJS: function toJS(val, fieldSettings) {
      return val;
    }
  }
});

var types = _objectSpread(_objectSpread({}, _basic["default"].types), {}, {
  number: _objectSpread(_objectSpread({}, _basic["default"].types.number), {}, {
    widgets: _objectSpread(_objectSpread({}, _basic["default"].types.number.widgets), {}, {
      rangeslider: {
        opProps: {
          between: {
            isSpecialRange: true
          },
          not_between: {
            isSpecialRange: true
          }
        },
        operators: ["between", "not_between", "is_empty", "is_not_empty"]
      }
    })
  }),
  date: _objectSpread(_objectSpread({}, _basic["default"].types.date), {}, {
    widgets: {
      date: _objectSpread(_objectSpread({}, _basic["default"].types.date.widgets.date), {}, {
        opProps: {
          between: {
            isSpecialRange: true
          },
          not_between: {
            isSpecialRange: true
          }
        }
      })
    }
  }),
  treeselect: {
    mainWidget: "treeselect",
    defaultOperator: "select_equals",
    widgets: {
      treeselect: {
        operators: ["select_equals", "select_not_equals"]
      },
      treemultiselect: {
        operators: ["select_any_in", "select_not_any_in"]
      }
    }
  },
  treemultiselect: {
    defaultOperator: "multiselect_equals",
    widgets: {
      treemultiselect: {
        operators: ["multiselect_equals", "multiselect_not_equals"]
      }
    }
  }
});

var _default = _objectSpread(_objectSpread({}, _basic["default"]), {}, {
  types: types,
  widgets: widgets,
  settings: settings
});

exports["default"] = _default;