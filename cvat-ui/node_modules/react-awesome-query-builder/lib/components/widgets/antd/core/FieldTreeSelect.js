"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireWildcard(require("react"));

var _antd = require("antd");

var _stuff = require("../../../../utils/stuff");

var _propTypes = _interopRequireDefault(require("prop-types"));

var _keys = _interopRequireDefault(require("lodash/keys"));

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

var FieldTreeSelect = /*#__PURE__*/function (_PureComponent) {
  _inherits(FieldTreeSelect, _PureComponent);

  var _super = _createSuper(FieldTreeSelect);

  function FieldTreeSelect(props) {
    var _this;

    _classCallCheck(this, FieldTreeSelect);

    _this = _super.call(this, props);

    _this.onChange = function (key) {
      _this.props.setField(key);
    };

    _this.filterTreeNode = function (input, option) {
      var dataForFilter = option;
      var keysForFilter = ["title", "value", "label", "altLabel", "fullLabel"];
      var valueForFilter = keysForFilter.map(function (k) {
        return typeof dataForFilter[k] == "string" ? dataForFilter[k] : "";
      }).join("\0");
      return valueForFilter.toLowerCase().indexOf(input.toLowerCase()) >= 0;
    };

    (0, _stuff.useOnPropsChanged)(_assertThisInitialized(_this));

    _this.onPropsChanged(props);

    return _this;
  }

  _createClass(FieldTreeSelect, [{
    key: "onPropsChanged",
    value: function onPropsChanged(props) {
      var items = props.items,
          fieldSeparator = props.config.settings.fieldSeparator;
      var optionsMaxWidth = 0;
      var initialOffset = 24; // arrow + checkbox for leftmost item

      var offset = 20;
      var padding = 5 * 2;
      this.treeData = this.getTreeData(items, function (_ref) {
        var label = _ref.label,
            path = _ref.path;
        optionsMaxWidth = Math.max(optionsMaxWidth, (0, _stuff.calcTextWidth)(label, null) + padding + (path.split(fieldSeparator).length - 1) * offset + initialOffset);
      });
      this.optionsMaxWidth = optionsMaxWidth;
    }
  }, {
    key: "getTreeData",
    value: function getTreeData(fields) {
      var _this2 = this;

      var fn = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : null;
      return (0, _keys["default"])(fields).map(function (fieldKey) {
        var field = fields[fieldKey];
        var items = field.items,
            key = field.key,
            path = field.path,
            label = field.label,
            fullLabel = field.fullLabel,
            altLabel = field.altLabel,
            tooltip = field.tooltip;
        if (fn) fn(field);

        var _path = path || key;

        var option = tooltip ? /*#__PURE__*/_react["default"].createElement(_antd.Tooltip, {
          title: tooltip
        }, label) : label;

        if (items) {
          return {
            value: _path,
            title: option,
            children: _this2.getTreeData(items, fn),
            selectable: false,
            altLabel: altLabel,
            fullLabel: fullLabel,
            label: label
          };
        } else {
          return {
            value: _path,
            title: option,
            altLabel: altLabel,
            fullLabel: fullLabel,
            label: label
          };
        }
      });
    }
  }, {
    key: "render",
    value: function render() {
      var _this$props = this.props,
          config = _this$props.config,
          _this$props$customPro = _this$props.customProps,
          customProps = _this$props$customPro === void 0 ? {} : _this$props$customPro,
          placeholder = _this$props.placeholder,
          selectedKey = _this$props.selectedKey,
          selectedLabel = _this$props.selectedLabel,
          selectedOpts = _this$props.selectedOpts,
          selectedAltLabel = _this$props.selectedAltLabel,
          selectedFullLabel = _this$props.selectedFullLabel,
          readonly = _this$props.readonly;
      var _config$settings = config.settings,
          renderSize = _config$settings.renderSize,
          fieldSeparator = _config$settings.fieldSeparator;
      var tooltipText = selectedAltLabel || selectedFullLabel;
      if (tooltipText == selectedLabel) tooltipText = null;
      var selectedPath = selectedKey ? selectedKey.split(fieldSeparator) : null;
      var treeDefaultExpandedKeys = selectedPath && selectedPath.length > 1 ? selectedPath.slice(0, -1).map(function (_key, i) {
        return selectedPath.slice(0, i + 1).join(fieldSeparator);
      }) : null;
      var placeholderWidth = (0, _stuff.calcTextWidth)(placeholder) + 6;
      var isFieldSelected = !!selectedKey;
      var minWidth = placeholderWidth + _stuff.SELECT_WIDTH_OFFSET_RIGHT;
      var dropdownMinWidth = 100;
      var dropdownMaxWidth = 800;
      var useAutoWidth = true; //tip: "auto" is good, but width will jump on expand/collapse

      var dropdownWidth = Math.max(dropdownMinWidth, Math.min(dropdownMaxWidth, this.optionsMaxWidth));

      var res = /*#__PURE__*/_react["default"].createElement(_antd.TreeSelect, _extends({
        onChange: this.onChange,
        value: selectedKey || undefined,
        style: {
          minWidth: minWidth,
          width: isFieldSelected ? null : minWidth
        },
        dropdownStyle: {
          width: useAutoWidth ? "auto" : dropdownWidth + 20,
          paddingRight: "10px"
        },
        multiple: false,
        treeCheckable: false,
        treeDataSimpleMode: false,
        treeData: this.treeData,
        size: renderSize,
        placeholder: placeholder,
        filterTreeNode: this.filterTreeNode,
        treeDefaultExpandedKeys: treeDefaultExpandedKeys,
        dropdownMatchSelectWidth: false,
        disabled: readonly
      }, customProps));

      if (tooltipText && !selectedOpts.tooltip) {
        res = /*#__PURE__*/_react["default"].createElement(_antd.Tooltip, {
          title: tooltipText
        }, res);
      }

      return res;
    }
  }]);

  return FieldTreeSelect;
}(_react.PureComponent);

exports["default"] = FieldTreeSelect;
FieldTreeSelect.propTypes = {
  config: _propTypes["default"].object.isRequired,
  customProps: _propTypes["default"].object,
  items: _propTypes["default"].array.isRequired,
  placeholder: _propTypes["default"].string,
  selectedKey: _propTypes["default"].string,
  selectedKeys: _propTypes["default"].array,
  selectedPath: _propTypes["default"].array,
  selectedLabel: _propTypes["default"].string,
  selectedAltLabel: _propTypes["default"].string,
  selectedFullLabel: _propTypes["default"].string,
  selectedOpts: _propTypes["default"].object,
  readonly: _propTypes["default"].bool,
  //actions
  setField: _propTypes["default"].func.isRequired
};