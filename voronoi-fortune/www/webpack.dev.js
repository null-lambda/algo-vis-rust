const CopyWebpackPlugin = require("copy-webpack-plugin");
const merge  = require("webpack-merge");
const common = require("./webpack.common.js");
const path = require("path");

module.exports = merge(common, {
  mode: "development",
});
