"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireWildcard(require("react"));

var _map = _interopRequireDefault(require("lodash/map"));

var _antd = require("antd");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

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

var ButtonGroup = _antd.Button.Group;

var ConjsButton = /*#__PURE__*/function (_PureComponent) {
  _inherits(ConjsButton, _PureComponent);

  var _super = _createSuper(ConjsButton);

  function ConjsButton() {
    var _this;

    _classCallCheck(this, ConjsButton);

    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }

    _this = _super.call.apply(_super, [this].concat(args));

    _this.onClick = function (_e) {
      var _this$props = _this.props,
          setConjunction = _this$props.setConjunction,
          item = _this$props.item;
      var conj = item.key;
      setConjunction(conj);
    };

    return _this;
  }

  _createClass(ConjsButton, [{
    key: "render",
    value: function render() {
      var _this$props2 = this.props,
          disabled = _this$props2.disabled,
          item = _this$props2.item;
      return /*#__PURE__*/_react["default"].createElement(_antd.Button, {
        disabled: disabled,
        type: item.checked ? "primary" : null,
        onClick: this.onClick
      }, item.label);
    }
  }]);

  return ConjsButton;
}(_react.PureComponent);

var ConjsButtons = /*#__PURE__*/function (_PureComponent2) {
  _inherits(ConjsButtons, _PureComponent2);

  var _super2 = _createSuper(ConjsButtons);

  function ConjsButtons() {
    var _this2;

    _classCallCheck(this, ConjsButtons);

    for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
      args[_key2] = arguments[_key2];
    }

    _this2 = _super2.call.apply(_super2, [this].concat(args));

    _this2.setNot = function (e) {
      var _this2$props = _this2.props,
          setNot = _this2$props.setNot,
          not = _this2$props.not;
      if (setNot) setNot(!not);
    };

    return _this2;
  }

  _createClass(ConjsButtons, [{
    key: "render",
    value: function render() {
      var _this$props3 = this.props,
          readonly = _this$props3.readonly,
          disabled = _this$props3.disabled,
          not = _this$props3.not,
          conjunctionOptions = _this$props3.conjunctionOptions,
          config = _this$props3.config,
          setConjunction = _this$props3.setConjunction;
      var conjsCount = Object.keys(conjunctionOptions).length;
      var lessThenTwo = disabled;
      return /*#__PURE__*/_react["default"].createElement(ButtonGroup, {
        key: "group-conjs-buttons",
        size: config.settings.renderSize,
        disabled: disabled || readonly
      }, config.settings.showNot && (readonly ? not : true) && /*#__PURE__*/_react["default"].createElement(_antd.Button, {
        key: "group-not",
        onClick: this.setNot,
        type: not ? "primary" : null,
        disabled: readonly
      }, config.settings.notLabel), conjsCount > 1 && !lessThenTwo && (0, _map["default"])(conjunctionOptions, function (item, _index) {
        return readonly && !item.checked ? null : /*#__PURE__*/_react["default"].createElement(ConjsButton, {
          key: item.id,
          item: item,
          disabled: disabled || readonly,
          setConjunction: setConjunction
        });
      }));
    }
  }]);

  return ConjsButtons;
}(_react.PureComponent); // obsolete

/*
const RadioButton = Radio.Button;
const RadioGroup = Radio.Group;
class ConjsRadios extends PureComponent {
  setConjunction = (e) => {
    const {setConjunction} = this.props;
    const conj = e.target.value;
    setConjunction(conj);
  }

  render() {
    const {readonly, disabled, selectedConjunction, conjunctionOptions, config} = this.props;
    return (
      <RadioGroup
        key="group-conjs-radios"
        disabled={disabled}
        value={selectedConjunction}
        size={config.settings.renderSize}
        onChange={this.setConjunction}
      >
        {map(conjunctionOptions, (item, _index) => readonly && !item.checked ? null : (
          <RadioButton
            key={item.id}
            value={item.key}
            //checked={item.checked}
          >{item.label}</RadioButton>
        ))}
      </RadioGroup>
    );
  }
}
*/


exports["default"] = ConjsButtons;