"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireWildcard(require("react"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _antd = require("antd");

var _stuff = require("../../../../utils/stuff");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _getRequireWildcardCache() { if (typeof WeakMap !== "function") return null; var cache = new WeakMap(); _getRequireWildcardCache = function _getRequireWildcardCache() { return cache; }; return cache; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } if (obj === null || _typeof(obj) !== "object" && typeof obj !== "function") { return { "default": obj }; } var cache = _getRequireWildcardCache(); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj["default"] = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

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

var TreeSelectWidget = /*#__PURE__*/function (_PureComponent) {
  _inherits(TreeSelectWidget, _PureComponent);

  var _super = _createSuper(TreeSelectWidget);

  function TreeSelectWidget(props) {
    var _this;

    _classCallCheck(this, TreeSelectWidget);

    _this = _super.call(this, props);

    _this.handleChange = function (val) {
      if (!_this.props.treeMultiple) {
        _this.props.setValue(val);

        return;
      }

      if (val && !val.length) {
        _this.props.setValue(undefined); //not allow []


        return;
      }

      if (_typeof(val[0]) == "object" && val[0].value !== undefined) {
        //`treeCheckStrictly` is on
        val = val.map(function (v) {
          return v.value;
        });
      }

      _this.props.setValue(val);
    };

    _this.filterTreeNode = function (input, option) {
      var dataForFilter = option.title;
      return dataForFilter.toLowerCase().indexOf(input.toLowerCase()) >= 0;
    };

    (0, _stuff.useOnPropsChanged)(_assertThisInitialized(_this));

    _this.onPropsChanged(props);

    return _this;
  }

  _createClass(TreeSelectWidget, [{
    key: "onPropsChanged",
    value: function onPropsChanged(props) {
      var listValues = props.listValues,
          treeMultiple = props.treeMultiple;
      var optionsMaxWidth = 0;
      var initialOffset = treeMultiple ? 24 + 22 : 24; // arrow + checkbox for leftmost item

      var offset = 20;
      var padding = 5 * 2;
      (0, _stuff.mapListValues)(listValues, function (_ref) {
        var title = _ref.title,
            value = _ref.value,
            path = _ref.path;
        optionsMaxWidth = Math.max(optionsMaxWidth, (0, _stuff.calcTextWidth)(title, null) + padding + (path ? path.length : 0) * offset + initialOffset);
      });
      this.optionsMaxWidth = optionsMaxWidth;
    }
  }, {
    key: "render",
    value: function render() {
      var _this$props = this.props,
          config = _this$props.config,
          placeholder = _this$props.placeholder,
          _this$props$customPro = _this$props.customProps,
          customProps = _this$props$customPro === void 0 ? {} : _this$props$customPro,
          value = _this$props.value,
          treeMultiple = _this$props.treeMultiple,
          listValues = _this$props.listValues,
          treeExpandAll = _this$props.treeExpandAll,
          readonly = _this$props.readonly;
      var treeCheckStrictly = customProps.treeCheckStrictly || false;
      var renderSize = config.settings.renderSize;
      var placeholderWidth = (0, _stuff.calcTextWidth)(placeholder) + 6;

      var _value = value != undefined ? value : undefined;

      if (treeCheckStrictly && _value !== undefined) {
        if (treeMultiple) {
          _value = _value.map(function (v) {
            return {
              value: v,
              label: (0, _stuff.getTitleInListValues)(listValues, v)
            };
          });
        }
      }

      var width = _value ? null : placeholderWidth + _stuff.SELECT_WIDTH_OFFSET_RIGHT;
      var dropdownMinWidth = 100;
      var dropdownMaxWidth = 800;
      var useAutoWidth = true; //tip: "auto" is good, but width will jump on expand/collapse

      var dropdownWidth = Math.max(dropdownMinWidth, Math.min(dropdownMaxWidth, this.optionsMaxWidth));
      return /*#__PURE__*/_react["default"].createElement(_antd.TreeSelect, _extends({
        disabled: readonly,
        style: {
          minWidth: width,
          width: width
        },
        dropdownStyle: {
          width: useAutoWidth ? "auto" : dropdownWidth + 10,
          paddingRight: "10px"
        },
        multiple: treeMultiple,
        treeCheckable: treeMultiple,
        key: "widget-treeselect",
        dropdownMatchSelectWidth: false,
        placeholder: placeholder,
        size: renderSize,
        treeData: listValues,
        treeDataSimpleMode: _stuff.defaultTreeDataMap,
        filterTreeNode: this.filterTreeNode,
        value: _value,
        onChange: this.handleChange,
        treeDefaultExpandAll: treeExpandAll
      }, customProps));
    }
  }]);

  return TreeSelectWidget;
}(_react.PureComponent);

exports["default"] = TreeSelectWidget;
TreeSelectWidget.propTypes = {
  setValue: _propTypes["default"].func.isRequired,
  config: _propTypes["default"].object.isRequired,
  value: _propTypes["default"].oneOfType([_propTypes["default"].string, _propTypes["default"].array]),
  field: _propTypes["default"].string.isRequired,
  placeholder: _propTypes["default"].string,
  customProps: _propTypes["default"].object,
  fieldDefinition: _propTypes["default"].object,
  readonly: _propTypes["default"].bool,
  treeMultiple: _propTypes["default"].bool,
  // from fieldSettings:
  listValues: _propTypes["default"].oneOfType([_propTypes["default"].object, _propTypes["default"].array])
};