"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.SqlString = void 0;

var SqlString = require("sqlstring");

exports.SqlString = SqlString;

SqlString.trim = function (val) {
  if (val.charAt(0) == "'") return val.substring(1, val.length - 1);else return val;
};

SqlString.escapeLike = function (val) {
  var any_start = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : true;
  var any_end = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : true;
  // normal escape
  var res = SqlString.escape(val); // unwrap ''

  res = SqlString.trim(res); // escape % and _

  res = res.replace(/[%_]/g, "\\$&"); // wrap with % for LIKE

  res = (any_start ? "%" : "") + res + (any_end ? "%" : ""); // wrap ''

  res = "'" + res + "'";
  return res;
};