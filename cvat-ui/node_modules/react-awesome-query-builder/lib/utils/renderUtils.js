"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.pureShouldComponentUpdate = exports.liteShouldComponentUpdate = void 0;

var _stuff = require("./stuff");

var liteShouldComponentUpdate = function liteShouldComponentUpdate(self, config) {
  return function (nextProps, nextState) {
    var prevProps = self.props;
    var prevState = self.state;
    var should = nextProps != prevProps || nextState != prevState;

    if (should) {
      if (prevState == nextState && prevProps != nextProps) {
        var chs = [];

        for (var k in nextProps) {
          var changed = nextProps[k] != prevProps[k];

          if (changed) {
            if (config[k] == "ignore") changed = false;else if (config[k] == "shallow_deep") changed = !(0, _stuff.shallowEqual)(nextProps[k], prevProps[k], true);else if (config[k] == "shallow") changed = !(0, _stuff.shallowEqual)(nextProps[k], prevProps[k]);else if (typeof config[k] == "function") changed = config[k](nextProps[k], prevProps[k], nextState);
          }

          if (changed) chs.push(k);
        }

        if (!chs.length) should = false;
      }
    }

    return should;
  };
};

exports.liteShouldComponentUpdate = liteShouldComponentUpdate;

var pureShouldComponentUpdate = function pureShouldComponentUpdate(self) {
  return function (nextProps, nextState) {
    return !(0, _stuff.shallowEqual)(self.props, nextProps) || !(0, _stuff.shallowEqual)(self.state, nextState);
  };
};

exports.pureShouldComponentUpdate = pureShouldComponentUpdate;