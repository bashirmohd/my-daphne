const express = require("express");
const path = require("path");
const app = express();
const axios = require("axios");
const querystring = require("querystring");
app.use(express.static(path.join(__dirname, "build")));

app.get("/", function(req, res) {
    res.sendFile(path.join(__dirname, "build", "index.html"));
});

app.get("/traffic", function(req, res) {
    const q = {
        query: `
    {
        mapTopology(name: "routed_toplevel") {
          edges {
            name
            netbeamTraffic
          }
        }
      }
    `
    };

    axios
        .get(`https://my.es.net/graphql?${querystring.stringify(q)}`)
        .then(function(response) {
            res.json(response.data);
        })
        .catch(err => {
            console.log(err);
        });
});

console.log("Running rodent servers");
app.listen(8080);
