"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = exports.FieldWrapper = void 0;

var _react = _interopRequireWildcard(require("react"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _RuleContainer = _interopRequireDefault(require("./containers/RuleContainer"));

var _Draggable = _interopRequireDefault(require("./containers/Draggable"));

var _Field = _interopRequireDefault(require("./Field"));

var _Operator = _interopRequireDefault(require("./Operator"));

var _Widget = _interopRequireDefault(require("./Widget"));

var _OperatorOptions = _interopRequireDefault(require("./OperatorOptions"));

var _configUtils = require("../utils/configUtils");

var _stuff = require("../utils/stuff");

var _dec, _class, _class2, _temp;

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _getRequireWildcardCache() { if (typeof WeakMap !== "function") return null; var cache = new WeakMap(); _getRequireWildcardCache = function _getRequireWildcardCache() { return cache; }; return cache; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } if (obj === null || _typeof(obj) !== "object" && typeof obj !== "function") { return { "default": obj }; } var cache = _getRequireWildcardCache(); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj["default"] = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { _defineProperty(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

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

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

function _objectWithoutProperties(source, excluded) { if (source == null) return {}; var target = _objectWithoutPropertiesLoose(source, excluded); var key, i; if (Object.getOwnPropertySymbols) { var sourceSymbolKeys = Object.getOwnPropertySymbols(source); for (i = 0; i < sourceSymbolKeys.length; i++) { key = sourceSymbolKeys[i]; if (excluded.indexOf(key) >= 0) continue; if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue; target[key] = source[key]; } } return target; }

function _objectWithoutPropertiesLoose(source, excluded) { if (source == null) return {}; var target = {}; var sourceKeys = Object.keys(source); var key, i; for (i = 0; i < sourceKeys.length; i++) { key = sourceKeys[i]; if (excluded.indexOf(key) >= 0) continue; target[key] = source[key]; } return target; }

var Col = function Col(_ref) {
  var children = _ref.children,
      props = _objectWithoutProperties(_ref, ["children"]);

  return /*#__PURE__*/_react["default"].createElement("div", props, children);
};

var dummyFn = function dummyFn() {};

var DragIcon = function DragIcon() {
  return /*#__PURE__*/_react["default"].createElement("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    viewBox: "0 0 24 24",
    fill: "gray",
    width: "18px",
    height: "18px"
  }, /*#__PURE__*/_react["default"].createElement("path", {
    d: "M0 0h24v24H0V0z",
    fill: "none"
  }), /*#__PURE__*/_react["default"].createElement("path", {
    d: "M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"
  }));
};

var ConfirmFn = function ConfirmFn(Cmp) {
  return function (props) {
    var useConfirm = props.config.settings.useConfirm;
    var confirmFn = useConfirm ? useConfirm() : null;
    return /*#__PURE__*/_react["default"].createElement(Cmp, _extends({}, props, {
      confirmFn: confirmFn
    }));
  };
};

var Rule = (_dec = (0, _Draggable["default"])("rule"), (0, _RuleContainer["default"])(_class = _dec(_class = ConfirmFn(_class = (_temp = _class2 = /*#__PURE__*/function (_PureComponent) {
  _inherits(Rule, _PureComponent);

  var _super = _createSuper(Rule);

  function Rule(props) {
    var _this;

    _classCallCheck(this, Rule);

    _this = _super.call(this, props);

    _this.removeSelf = function () {
      var confirmFn = _this.props.confirmFn;
      var _this$props$config$se = _this.props.config.settings,
          renderConfirm = _this$props$config$se.renderConfirm,
          confirmOptions = _this$props$config$se.removeRuleConfirmOptions;

      var doRemove = function doRemove() {
        _this.props.removeSelf();
      };

      if (confirmOptions && !_this.isEmptyCurrentRule()) {
        renderConfirm(_objectSpread(_objectSpread({}, confirmOptions), {}, {
          onOk: doRemove,
          onCancel: null,
          confirmFn: confirmFn
        }));
      } else {
        doRemove();
      }
    };

    _this.isEmptyCurrentRule = function () {
      return !(_this.props.selectedField !== null && _this.props.selectedOperator !== null && _this.props.value.filter(function (val) {
        return val !== undefined;
      }).size > 0);
    };

    (0, _stuff.useOnPropsChanged)(_assertThisInitialized(_this));

    _this.onPropsChanged(props);

    return _this;
  }

  _createClass(Rule, [{
    key: "onPropsChanged",
    value: function onPropsChanged(nextProps) {
      var prevProps = this.props;
      var keysForMeta = ["selectedField", "selectedOperator", "config", "reordableNodesCnt"];
      var needUpdateMeta = !this.meta || keysForMeta.map(function (k) {
        return nextProps[k] !== prevProps[k];
      }).filter(function (ch) {
        return ch;
      }).length > 0;

      if (needUpdateMeta) {
        this.meta = this.getMeta(nextProps);
      }
    }
  }, {
    key: "getMeta",
    value: function getMeta(_ref2) {
      var selectedField = _ref2.selectedField,
          selectedOperator = _ref2.selectedOperator,
          config = _ref2.config,
          reordableNodesCnt = _ref2.reordableNodesCnt;
      var selectedFieldPartsLabels = (0, _configUtils.getFieldPathLabels)(selectedField, config);
      var selectedFieldConfig = (0, _configUtils.getFieldConfig)(selectedField, config);
      var isSelectedGroup = selectedFieldConfig && selectedFieldConfig.type == "!struct";
      var isFieldAndOpSelected = selectedField && selectedOperator && !isSelectedGroup;
      var selectedOperatorConfig = (0, _configUtils.getOperatorConfig)(config, selectedOperator, selectedField);
      var selectedOperatorHasOptions = selectedOperatorConfig && selectedOperatorConfig.options != null;
      var selectedFieldWidgetConfig = (0, _configUtils.getFieldWidgetConfig)(config, selectedField, selectedOperator) || {};
      var isOnlyValue = selectedField && selectedFieldConfig.valueSources.length == 1 && selectedFieldConfig.valueSources[0] == "value";
      var hideOperator = selectedFieldWidgetConfig.hideOperator && isOnlyValue;
      var showDragIcon = config.settings.canReorder && reordableNodesCnt > 1;
      var showOperator = selectedField && !hideOperator;
      var showOperatorLabel = selectedField && hideOperator && selectedFieldWidgetConfig.operatorInlineLabel;
      var showWidget = isFieldAndOpSelected;
      var showOperatorOptions = isFieldAndOpSelected && selectedOperatorHasOptions;
      return {
        selectedFieldPartsLabels: selectedFieldPartsLabels,
        selectedFieldWidgetConfig: selectedFieldWidgetConfig,
        showDragIcon: showDragIcon,
        showOperator: showOperator,
        showOperatorLabel: showOperatorLabel,
        showWidget: showWidget,
        showOperatorOptions: showOperatorOptions
      };
    }
  }, {
    key: "render",
    value: function render() {
      var _this$props = this.props,
          config = _this$props.config,
          valueError = _this$props.valueError;
      var _this$meta = this.meta,
          selectedFieldPartsLabels = _this$meta.selectedFieldPartsLabels,
          selectedFieldWidgetConfig = _this$meta.selectedFieldWidgetConfig,
          showDragIcon = _this$meta.showDragIcon,
          showOperator = _this$meta.showOperator,
          showOperatorLabel = _this$meta.showOperatorLabel,
          showWidget = _this$meta.showWidget,
          showOperatorOptions = _this$meta.showOperatorOptions;
      var _config$settings = config.settings,
          deleteLabel = _config$settings.deleteLabel,
          renderBeforeWidget = _config$settings.renderBeforeWidget,
          renderAfterWidget = _config$settings.renderAfterWidget,
          renderSize = _config$settings.renderSize,
          immutableGroupsMode = _config$settings.immutableGroupsMode,
          immutableFieldsMode = _config$settings.immutableFieldsMode,
          immutableOpsMode = _config$settings.immutableOpsMode,
          immutableValuesMode = _config$settings.immutableValuesMode,
          renderRuleError = _config$settings.renderRuleError,
          showErrorMessage = _config$settings.showErrorMessage,
          Btn = _config$settings.renderButton;

      var field = /*#__PURE__*/_react["default"].createElement(FieldWrapper, {
        key: "field",
        classname: "rule--field",
        config: config,
        selectedField: this.props.selectedField,
        setField: !immutableOpsMode ? this.props.setField : dummyFn,
        parentField: this.props.parentField,
        readonly: immutableFieldsMode
      });

      var operator = /*#__PURE__*/_react["default"].createElement(OperatorWrapper, {
        key: "operator",
        config: config,
        selectedField: this.props.selectedField,
        selectedOperator: this.props.selectedOperator,
        setOperator: !immutableOpsMode ? this.props.setOperator : dummyFn,
        selectedFieldPartsLabels: selectedFieldPartsLabels,
        showOperator: showOperator,
        showOperatorLabel: showOperatorLabel,
        selectedFieldWidgetConfig: selectedFieldWidgetConfig,
        readonly: immutableOpsMode
      });

      var widget = showWidget && /*#__PURE__*/_react["default"].createElement(Col, {
        key: "widget-for-" + this.props.selectedOperator,
        className: "rule--value"
      }, /*#__PURE__*/_react["default"].createElement(_Widget["default"], {
        key: "values",
        field: this.props.selectedField,
        operator: this.props.selectedOperator,
        value: this.props.value,
        valueSrc: this.props.valueSrc,
        valueError: valueError,
        config: config,
        setValue: !immutableValuesMode ? this.props.setValue : dummyFn,
        setValueSrc: !immutableValuesMode ? this.props.setValueSrc : dummyFn,
        readonly: immutableValuesMode
      }));

      var operatorOptions = showOperatorOptions && /*#__PURE__*/_react["default"].createElement(Col, {
        key: "op-options-for-" + this.props.selectedOperator,
        className: "rule--operator-options"
      }, /*#__PURE__*/_react["default"].createElement(_OperatorOptions["default"], {
        key: "operatorOptions",
        selectedField: this.props.selectedField,
        selectedOperator: this.props.selectedOperator,
        operatorOptions: this.props.operatorOptions,
        setOperatorOption: !immutableOpsMode ? this.props.setOperatorOption : dummyFn,
        config: config,
        readonly: immutableValuesMode
      }));

      var beforeWidget = renderBeforeWidget && /*#__PURE__*/_react["default"].createElement(Col, {
        key: "before-widget-for-" + this.props.selectedOperator,
        className: "rule--before-widget"
      }, typeof renderBeforeWidget === "function" ? renderBeforeWidget(this.props) : renderBeforeWidget);

      var afterWidget = renderAfterWidget && /*#__PURE__*/_react["default"].createElement(Col, {
        key: "after-widget-for-" + this.props.selectedOperator,
        className: "rule--after-widget"
      }, typeof renderAfterWidget === "function" ? renderAfterWidget(this.props) : renderAfterWidget);

      var oneValueError = valueError && valueError.toArray().filter(function (e) {
        return !!e;
      }).shift() || null;

      var error = showErrorMessage && oneValueError && /*#__PURE__*/_react["default"].createElement("div", {
        className: "rule--error"
      }, renderRuleError ? renderRuleError({
        error: oneValueError
      }) : oneValueError);

      var parts = [field, operator, beforeWidget, widget, afterWidget, operatorOptions];

      var drag = showDragIcon && /*#__PURE__*/_react["default"].createElement("span", {
        key: "rule-drag-icon",
        className: "qb-drag-handler rule--drag-handler",
        onMouseDown: this.props.handleDraggerMouseDown
      }, /*#__PURE__*/_react["default"].createElement(DragIcon, null), " ");

      var del = /*#__PURE__*/_react["default"].createElement("div", {
        key: "rule-header",
        className: "rule--header"
      }, !immutableGroupsMode && /*#__PURE__*/_react["default"].createElement(Btn, {
        type: "delRule",
        onClick: this.removeSelf,
        label: deleteLabel,
        config: config
      }));

      var body = /*#__PURE__*/_react["default"].createElement("div", {
        key: "rule-body",
        className: "rule--body"
      }, parts);

      return /*#__PURE__*/_react["default"].createElement(_react["default"].Fragment, null, drag, /*#__PURE__*/_react["default"].createElement("div", {
        className: "rule--body--wrapper"
      }, body, error), del);
    }
  }]);

  return Rule;
}(_react.PureComponent), _class2.propTypes = {
  selectedField: _propTypes["default"].string,
  selectedOperator: _propTypes["default"].string,
  operatorOptions: _propTypes["default"].object,
  config: _propTypes["default"].object.isRequired,
  value: _propTypes["default"].any,
  //depends on widget
  valueSrc: _propTypes["default"].any,
  isDraggingMe: _propTypes["default"].bool,
  isDraggingTempo: _propTypes["default"].bool,
  parentField: _propTypes["default"].string,
  //from RuleGroup
  valueError: _propTypes["default"].any,
  //path: PropTypes.instanceOf(Immutable.List),
  //actions
  handleDraggerMouseDown: _propTypes["default"].func,
  setField: _propTypes["default"].func,
  setOperator: _propTypes["default"].func,
  setOperatorOption: _propTypes["default"].func,
  removeSelf: _propTypes["default"].func,
  setValue: _propTypes["default"].func,
  setValueSrc: _propTypes["default"].func,
  reordableNodesCnt: _propTypes["default"].number
}, _temp)) || _class) || _class) || _class);

var FieldWrapper = /*#__PURE__*/function (_PureComponent2) {
  _inherits(FieldWrapper, _PureComponent2);

  var _super2 = _createSuper(FieldWrapper);

  function FieldWrapper() {
    _classCallCheck(this, FieldWrapper);

    return _super2.apply(this, arguments);
  }

  _createClass(FieldWrapper, [{
    key: "render",
    value: function render() {
      var _this$props2 = this.props,
          config = _this$props2.config,
          selectedField = _this$props2.selectedField,
          setField = _this$props2.setField,
          parentField = _this$props2.parentField,
          classname = _this$props2.classname,
          readonly = _this$props2.readonly;
      return /*#__PURE__*/_react["default"].createElement(Col, {
        className: classname
      }, config.settings.showLabels && /*#__PURE__*/_react["default"].createElement("label", {
        className: "rule--label"
      }, config.settings.fieldLabel), /*#__PURE__*/_react["default"].createElement(_Field["default"], {
        config: config,
        selectedField: selectedField,
        parentField: parentField,
        setField: setField,
        customProps: config.settings.customFieldSelectProps,
        readonly: readonly
      }));
    }
  }]);

  return FieldWrapper;
}(_react.PureComponent);

exports.FieldWrapper = FieldWrapper;

var OperatorWrapper = /*#__PURE__*/function (_PureComponent3) {
  _inherits(OperatorWrapper, _PureComponent3);

  var _super3 = _createSuper(OperatorWrapper);

  function OperatorWrapper() {
    _classCallCheck(this, OperatorWrapper);

    return _super3.apply(this, arguments);
  }

  _createClass(OperatorWrapper, [{
    key: "render",
    value: function render() {
      var _this$props3 = this.props,
          config = _this$props3.config,
          selectedField = _this$props3.selectedField,
          selectedOperator = _this$props3.selectedOperator,
          setOperator = _this$props3.setOperator,
          selectedFieldPartsLabels = _this$props3.selectedFieldPartsLabels,
          showOperator = _this$props3.showOperator,
          showOperatorLabel = _this$props3.showOperatorLabel,
          selectedFieldWidgetConfig = _this$props3.selectedFieldWidgetConfig,
          readonly = _this$props3.readonly;

      var operator = showOperator && /*#__PURE__*/_react["default"].createElement(Col, {
        key: "operators-for-" + (selectedFieldPartsLabels || []).join("_"),
        className: "rule--operator"
      }, config.settings.showLabels && /*#__PURE__*/_react["default"].createElement("label", {
        className: "rule--label"
      }, config.settings.operatorLabel), /*#__PURE__*/_react["default"].createElement(_Operator["default"], {
        key: "operator",
        config: config,
        selectedField: selectedField,
        selectedOperator: selectedOperator,
        setOperator: setOperator,
        readonly: readonly
      }));

      var hiddenOperator = showOperatorLabel && /*#__PURE__*/_react["default"].createElement(Col, {
        key: "operators-for-" + (selectedFieldPartsLabels || []).join("_"),
        className: "rule--operator"
      }, /*#__PURE__*/_react["default"].createElement("div", {
        className: "rule--operator"
      }, config.settings.showLabels ? /*#__PURE__*/_react["default"].createElement("label", {
        className: "rule--label"
      }, "\xA0") : null, /*#__PURE__*/_react["default"].createElement("span", null, selectedFieldWidgetConfig.operatorInlineLabel)));

      return [operator, hiddenOperator];
    }
  }]);

  return OperatorWrapper;
}(_react.PureComponent);

var _default = Rule;
exports["default"] = _default;