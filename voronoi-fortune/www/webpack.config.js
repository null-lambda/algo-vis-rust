const CopyWebpackPlugin = require("copy-webpack-plugin");
const path = require('path');

module.exports = {
  entry: "./bootstrap.js",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bootstrap.js",
  },
  mode: "development",
  plugins: [
    new CopyWebpackPlugin(['index.html'])
  ],
  module: {
    rules: [
      {
        test: /\.scss\.raw$/i,
        use: ["raw-loader", "sass-loader"],
      },
      {
        test: /\.scss$/i, 
        use: ["style-loader", "css-loader", "sass-loader"],
      }
    ],
  },
 experiments: {
    asyncWebAssembly: true,
    topLevelAwait: true,
  },
};
