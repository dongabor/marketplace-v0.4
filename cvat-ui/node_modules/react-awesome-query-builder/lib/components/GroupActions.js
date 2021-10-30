"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GroupActions = void 0;

var _react = _interopRequireWildcard(require("react"));

function _getRequireWildcardCache() { if (typeof WeakMap !== "function") return null; var cache = new WeakMap(); _getRequireWildcardCache = function _getRequireWildcardCache() { return cache; }; return cache; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } if (obj === null || _typeof(obj) !== "object" && typeof obj !== "function") { return { "default": obj }; } var cache = _getRequireWildcardCache(); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj["default"] = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

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

var groupActionsPositionList = {
  topLeft: "group--actions--tl",
  topCenter: "group--actions--tc",
  topRight: "group--actions--tr",
  bottomLeft: "group--actions--bl",
  bottomCenter: "group--actions--bc",
  bottomRight: "group--actions--br"
};
var defaultPosition = "topRight";

var GroupActions = /*#__PURE__*/function (_PureComponent) {
  _inherits(GroupActions, _PureComponent);

  var _super = _createSuper(GroupActions);

  function GroupActions() {
    _classCallCheck(this, GroupActions);

    return _super.apply(this, arguments);
  }

  _createClass(GroupActions, [{
    key: "render",
    value: function render() {
      var _this$props = this.props,
          config = _this$props.config,
          addRule = _this$props.addRule,
          addGroup = _this$props.addGroup,
          canAddGroup = _this$props.canAddGroup,
          canAddRule = _this$props.canAddRule,
          canDeleteGroup = _this$props.canDeleteGroup,
          removeSelf = _this$props.removeSelf;
      var _config$settings = config.settings,
          immutableGroupsMode = _config$settings.immutableGroupsMode,
          addRuleLabel = _config$settings.addRuleLabel,
          addGroupLabel = _config$settings.addGroupLabel,
          delGroupLabel = _config$settings.delGroupLabel,
          groupActionsPosition = _config$settings.groupActionsPosition,
          Btn = _config$settings.renderButton,
          BtnGrp = _config$settings.renderButtonGroup;
      var position = groupActionsPositionList[groupActionsPosition || defaultPosition];

      var addRuleBtn = !immutableGroupsMode && canAddRule && /*#__PURE__*/_react["default"].createElement(Btn, {
        type: "addRule",
        onClick: addRule,
        label: addRuleLabel,
        config: config
      });

      var addGroupBtn = !immutableGroupsMode && canAddGroup && /*#__PURE__*/_react["default"].createElement(Btn, {
        type: "addGroup",
        onClick: addGroup,
        label: addGroupLabel,
        config: config
      });

      var delGroupBtn = !immutableGroupsMode && canDeleteGroup && /*#__PURE__*/_react["default"].createElement(Btn, {
        type: "delGroup",
        onClick: removeSelf,
        label: delGroupLabel,
        config: config
      });

      return /*#__PURE__*/_react["default"].createElement("div", {
        className: "group--actions ".concat(position)
      }, /*#__PURE__*/_react["default"].createElement(BtnGrp, {
        config: config
      }, addRuleBtn, addGroupBtn, delGroupBtn));
    }
  }]);

  return GroupActions;
}(_react.PureComponent);

exports.GroupActions = GroupActions;