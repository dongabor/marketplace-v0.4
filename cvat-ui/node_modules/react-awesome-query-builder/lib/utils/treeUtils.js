"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.getLightTree = exports.getTreeBadFields = exports.getTotalRulesCountInTree = exports.getTotalReordableNodesCountInTree = exports.getFlatTree = exports.fixEmptyGroupsInTree = exports.fixPathsInTree = exports.removePathsInTree = exports.getItemByPath = exports.expandTreeSubpath = exports.expandTreePath = void 0;

var _immutable = _interopRequireDefault(require("immutable"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

/**
 * @param {Immutable.List} path
 * @param {...string} suffix
 * @return {Immutable.List}
 */
var expandTreePath = function expandTreePath(path) {
  for (var _len = arguments.length, suffix = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    suffix[_key - 1] = arguments[_key];
  }

  return path.interpose("children1").withMutations(function (list) {
    list.skip(1);
    list.push.apply(list, suffix);
    return list;
  });
};
/**
 * @param {Immutable.List} path
 * @param {...string} suffix
 * @return {Immutable.List}
 */


exports.expandTreePath = expandTreePath;

var expandTreeSubpath = function expandTreeSubpath(path) {
  for (var _len2 = arguments.length, suffix = new Array(_len2 > 1 ? _len2 - 1 : 0), _key2 = 1; _key2 < _len2; _key2++) {
    suffix[_key2 - 1] = arguments[_key2];
  }

  return path.interpose("children1").withMutations(function (list) {
    list.push.apply(list, suffix);
    return list;
  });
};
/**
 * @param {Immutable.Map} path
 * @param {Immutable.List} path
 * @return {Immutable.Map}
 */


exports.expandTreeSubpath = expandTreeSubpath;

var getItemByPath = function getItemByPath(tree, path) {
  var children = new _immutable["default"].OrderedMap(_defineProperty({}, tree.get("id"), tree));
  var res = tree;
  path.forEach(function (id) {
    res = children.get(id);
    children = res.get("children1");
  });
  return res;
};
/**
 * Remove `path` in every item
 * @param {Immutable.Map} tree
 * @return {Immutable.Map} tree
 */


exports.getItemByPath = getItemByPath;

var removePathsInTree = function removePathsInTree(tree) {
  var newTree = tree;

  function _processNode(item, path) {
    var itemPath = path.push(item.get("id"));

    if (item.get("path")) {
      newTree = newTree.removeIn(expandTreePath(itemPath, "path"));
    }

    var children = item.get("children1");

    if (children) {
      children.map(function (child, _childId) {
        _processNode(child, itemPath);
      });
    }
  }

  _processNode(tree, new _immutable["default"].List());

  return newTree;
};
/**
 * Set correct `path` in every item
 * @param {Immutable.Map} tree
 * @return {Immutable.Map} tree
 */


exports.removePathsInTree = removePathsInTree;

var fixPathsInTree = function fixPathsInTree(tree) {
  var newTree = tree;

  function _processNode(item, path, lev) {
    if (!item) return;

    var _id = item.get("id");

    var itemPath = path.push(item.get("id"));
    var currItemPath = item.get("path");

    if (!currItemPath || !currItemPath.equals(itemPath)) {
      newTree = newTree.setIn(expandTreePath(itemPath, "path"), itemPath);
    }

    var children = item.get("children1");

    if (children) {
      children.map(function (child, _childId) {
        _processNode(child, itemPath, lev + 1);
      });
    }
  }

  _processNode(tree, new _immutable["default"].List(), 0);

  return newTree;
};

exports.fixPathsInTree = fixPathsInTree;

var fixEmptyGroupsInTree = function fixEmptyGroupsInTree(tree) {
  var newTree = tree;

  function _processNode(item, path, lev) {
    if (!item) return false;
    var id = item.get("id");
    var itemPath = path.push(item.get("id"));
    var children = item.get("children1");

    if (children) {
      var allChildrenGone = children.map(function (child, _childId) {
        return _processNode(child, itemPath, lev + 1);
      }).reduce(function (curr, v) {
        return curr && v;
      }, true);

      if ((children.size == 0 || allChildrenGone) && lev > 0) {
        newTree = newTree.deleteIn(expandTreePath(itemPath));
        return true;
      }
    }

    return false;
  }

  _processNode(tree, new _immutable["default"].List(), 0);

  return newTree;
};
/**
 * @param {Immutable.Map} tree
 * @return {Object} {flat, items}
 */


exports.fixEmptyGroupsInTree = fixEmptyGroupsInTree;

var getFlatTree = function getFlatTree(tree) {
  var flat = [];
  var items = {};
  var realHeight = 0;

  function _flatizeTree(item, path, insideCollapsed, lev, info, parentType) {
    var type = item.get("type");
    var collapsed = item.get("collapsed");
    var id = item.get("id");
    var children = item.get("children1");
    var childrenIds = children ? children.map(function (_child, childId) {
      return childId;
    }) : null;
    var itemsBefore = flat.length;
    var top = realHeight;
    flat.push(id);
    if (!insideCollapsed) realHeight += 1;
    info.height = (info.height || 0) + 1;

    if (children) {
      var subinfo = {};
      children.map(function (child, _childId) {
        _flatizeTree(child, path.concat(id), insideCollapsed || collapsed, lev + 1, subinfo, type);
      });

      if (!collapsed) {
        info.height = (info.height || 0) + (subinfo.height || 0);
      }
    }

    var itemsAfter = flat.length;
    var _bottom = realHeight;
    var height = info.height;
    items[id] = {
      type: type,
      parent: path.length ? path[path.length - 1] : null,
      parentType: parentType,
      path: path.concat(id),
      lev: lev,
      leaf: !children,
      index: itemsBefore,
      id: id,
      children: childrenIds,
      _top: itemsBefore,
      _height: itemsAfter - itemsBefore,
      top: insideCollapsed ? null : top,
      height: height,
      bottom: (insideCollapsed ? null : top) + height,
      collapsed: collapsed,
      node: item
    };
  }

  _flatizeTree(tree, [], false, 0, {}, null);

  for (var i = 0; i < flat.length; i++) {
    var prevId = i > 0 ? flat[i - 1] : null;
    var nextId = i < flat.length - 1 ? flat[i + 1] : null;
    var item = items[flat[i]];
    item.prev = prevId;
    item.next = nextId;
  }

  return {
    flat: flat,
    items: items
  };
};
/**
 * Returns count of reorderable(!) nodes
 * @param {Immutable.Map} tree
 * @return {Integer}
 */


exports.getFlatTree = getFlatTree;

var getTotalReordableNodesCountInTree = function getTotalReordableNodesCountInTree(tree) {
  if (!tree) return -1;
  var cnt = 0;

  function _processNode(item, path, lev) {
    var id = item.get("id");
    var children = item.get("children1");
    var isRuleGroup = item.get("type") == "rule_group";
    cnt++; //tip: rules in rule-group can be reordered only inside

    if (children && !isRuleGroup) {
      children.map(function (child, _childId) {
        _processNode(child, path.concat(id), lev + 1);
      });
    }
  }

  _processNode(tree, [], 0);

  return cnt - 1; // -1 for root
};
/**
 * Returns count of rules (leafs, i.e. don't count groups)
 * @param {Immutable.Map} tree
 * @return {Integer}
 */


exports.getTotalReordableNodesCountInTree = getTotalReordableNodesCountInTree;

var getTotalRulesCountInTree = function getTotalRulesCountInTree(tree) {
  if (!tree) return -1;
  var cnt = 0;

  function _processNode(item, path, lev) {
    var id = item.get("id");
    var children = item.get("children1");
    var isGroup = item.get("type") == "group"; //const isRuleGroup = item.get("type") == "rule_group";

    if (children && isGroup) {
      children.map(function (child, _childId) {
        _processNode(child, path.concat(id), lev + 1);
      });
    } else {
      // tip: count rule_group as 1 rule
      cnt++;
    }
  }

  _processNode(tree, [], 0);

  return cnt;
};

exports.getTotalRulesCountInTree = getTotalRulesCountInTree;

var getTreeBadFields = function getTreeBadFields(tree) {
  var badFields = [];

  function _processNode(item, path, lev) {
    var id = item.get("id");
    var children = item.get("children1");
    var valueError = item.getIn(["properties", "valueError"]);
    var field = item.getIn(["properties", "field"]);

    if (valueError && valueError.size > 0 && valueError.filter(function (v) {
      return v != null;
    }).size > 0) {
      badFields.push(field);
    }

    if (children) {
      children.map(function (child, _childId) {
        _processNode(child, path.concat(id), lev + 1);
      });
    }
  }

  if (tree) _processNode(tree, [], 0);
  return Array.from(new Set(badFields));
}; // Remove fields that can be calced: "id", "path"
// Remove empty fields: "operatorOptions"


exports.getTreeBadFields = getTreeBadFields;

var getLightTree = function getLightTree(tree) {
  var newTree = tree;

  function _processNode(item, itemId) {
    if (item.path) delete item.path;
    if (itemId) delete item.id;
    var properties = item.properties;

    if (properties) {
      if (properties.operatorOptions == null) delete properties.operatorOptions;
    }

    var children = item.children1;

    if (children) {
      for (var id in children) {
        _processNode(children[id], id);
      }
    }
  }

  _processNode(tree, null);

  return newTree;
};

exports.getLightTree = getLightTree;