import React, { Component } from "react";

import loader from "../img/loader.gif";

export default class Spinner extends Component {
    render() {
        return (
            <div>
                <img
                    className="img-responsive"
                    style={{ margin: "0 auto", paddingTop: "50px" }}
                    src={loader}
                    alt="Item loading..."
                />
            </div>
        );
    }
}
