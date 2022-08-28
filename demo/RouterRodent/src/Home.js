import React from "react";
import { Row, Col } from "react-bootstrap";

const games = [
    {
        name: "ornl-nersc",
        path: ["ORNL", "ATLA", "NASH", "HOUS", "ELPA", "SUNN", "NERSC"]
    },
    {
        name: "nersc-fnal",
        path: ["NERSC", "SUNN", "SACR", "DENV", "KANS", "CHIC", "STAR", "FNAL"]
    },
    {
        name: "ornl-anl",
        path: ["ORNL", "ATLA", "WASH", "CHIC", "STAR", "ANL"]
    },
    {
        name: "fnal-cern",
        path: ["FNAL", "STAR", "BOST", "AMST", "CERN-513", "CERN"]
    },
    {
        name: "lbnl-cern",
        path: [
            "LBNL",
            "SUNN",
            "SACR",
            "DENV",
            "KANS",
            "CHIC",
            "WASH",
            "CERN-513",
            "CERN"
        ]
    },
    // {
    //     name: "llnl-ornl",
    //     path: ["LLNL", "SUNN", "ELPA", "HOUS", "NASH", "ATLA", "ORNL"]
    // }
    {
        name: "demo",
        path: ["LBNL", "SUNN", "ELPA"]
    }
];

export default class Home extends React.Component {
    render() {
        return (
            <div className="App">
                <Col style={{ backgroundColor: "#282c34" }}>
                    <div style={{ background: "#282c34" }}>
                        <Row>
                            <Col>
                                <button
                                    type="button"
                                    className="font btn btn-secondary"
                                    onClick={() =>
                                        this.props.onbeginGame(
                                            games[0].name,
                                            games[0].path
                                        )
                                    }
                                    style={{
                                        marginTop: 20,
                                        marginBottom: 0
                                    }}
                                >
                                    <pre
                                        style={{
                                            marginTop: 20,
                                            marginBottom: 20,
                                            marginLeft: 20,
                                            marginRight: 20
                                        }}
                                    >
                                        <h3
                                            style={{
                                                marginTop: 20,
                                                marginBottom: 20,
                                                marginLeft: 20,
                                                marginRight: 20
                                            }}
                                        >
                                            {games[0].name}
                                        </h3>
                                    </pre>
                                </button>
                            </Col>
                            <Col>
                                <button
                                    type="button"
                                    className="font btn btn-secondary"
                                    onClick={() =>
                                        this.props.onbeginGame(
                                            games[1].name,
                                            games[1].path
                                        )
                                    }
                                    style={{
                                        marginTop: 20,
                                        marginBottom: 0
                                    }}
                                >
                                    <pre
                                        style={{
                                            marginTop: 20,
                                            marginBottom: 20,
                                            marginLeft: 20,
                                            marginRight: 20
                                        }}
                                    >
                                        <h3
                                            style={{
                                                marginTop: 20,
                                                marginBottom: 20,
                                                marginLeft: 20,
                                                marginRight: 20
                                            }}
                                        >
                                            {games[1].name}
                                        </h3>
                                    </pre>
                                </button>
                            </Col>
                        </Row>
                        <Row>
                            <Col>
                                <button
                                    type="button"
                                    className="font btn btn-secondary"
                                    onClick={() =>
                                        this.props.onbeginGame(
                                            games[2].name,
                                            games[2].path
                                        )
                                    }
                                    style={{
                                        marginTop: 20,
                                        marginBottom: 0
                                    }}
                                >
                                    <pre
                                        style={{
                                            marginTop: 20,
                                            marginBottom: 20,
                                            marginLeft: 36,
                                            marginRight: 36
                                        }}
                                    >
                                        <h3
                                            style={{
                                                marginTop: 20,
                                                marginBottom: 20,
                                                marginLeft: 20,
                                                marginRight: 20
                                            }}
                                        >
                                            {games[2].name}
                                        </h3>
                                    </pre>
                                </button>
                            </Col>
                            <Col>
                                <button
                                    type="button"
                                    className="font btn btn-secondary"
                                    onClick={() =>
                                        this.props.onbeginGame(
                                            games[3].name,
                                            games[3].path
                                        )
                                    }
                                    style={{
                                        marginTop: 20,
                                        marginBottom: 0
                                    }}
                                >
                                    <pre
                                        style={{
                                            marginTop: 20,
                                            marginBottom: 20,
                                            marginLeft: 27,
                                            marginRight: 27
                                        }}
                                    >
                                        <h3
                                            style={{
                                                marginTop: 20,
                                                marginBottom: 20,
                                                marginLeft: 20,
                                                marginRight: 20
                                            }}
                                        >
                                            {games[3].name}
                                        </h3>
                                    </pre>
                                </button>
                            </Col>
                        </Row>
                        <Row>
                            <Col>
                                <button
                                    type="button"
                                    className="font btn btn-secondary"
                                    onClick={() =>
                                        this.props.onbeginGame(
                                            games[4].name,
                                            games[4].path
                                        )
                                    }
                                    style={{
                                        marginTop: 20,
                                        marginBottom: 0
                                    }}
                                >
                                    <pre
                                        style={{
                                            marginTop: 20,
                                            marginBottom: 20,
                                            marginLeft: 27,
                                            marginRight: 27
                                        }}
                                    >
                                        <h3
                                            style={{
                                                marginTop: 20,
                                                marginBottom: 20,
                                                marginLeft: 20,
                                                marginRight: 20
                                            }}
                                        >
                                            {games[4].name}
                                        </h3>
                                    </pre>
                                </button>
                            </Col>
                            {/* <Col>
                                <button
                                    type="button"
                                    className="font btn btn-secondary"
                                    onClick={() =>
                                        this.props.onbeginGame(
                                            games[5].name,
                                            games[5].path
                                        )
                                    }
                                    style={{
                                        marginTop: 20,
                                        marginBottom: 0
                                    }}
                                >
                                    <pre
                                        style={{
                                            marginTop: 20,
                                            marginBottom: 20,
                                            marginLeft: 20,
                                            marginRight: 20
                                        }}
                                    >
                                        <h3
                                            style={{
                                                marginTop: 20,
                                                marginBottom: 20,
                                                marginLeft: 20,
                                                marginRight: 20
                                            }}
                                        >
                                            {games[5].name}
                                        </h3>
                                    </pre>
                                </button>
                            </Col> */}
                            <Col>
                                <button
                                    type="button"
                                    className="font btn btn-secondary"
                                    onClick={() =>
                                        this.props.onbeginGame(
                                            games[5].name,
                                            games[5].path
                                        )
                                    }
                                    style={{
                                        marginTop: 20,
                                        marginBottom: 0
                                    }}
                                >
                                    <pre
                                        style={{
                                            marginTop: 20,
                                            marginBottom: 20,
                                            marginLeft: 68,
                                            marginRight: 68
                                        }}
                                    >
                                        <h3
                                            style={{
                                                marginTop: 20,
                                                marginBottom: 20,
                                                marginLeft: 20,
                                                marginRight: 20
                                            }}
                                        >
                                            {games[5].name}
                                        </h3>
                                    </pre>
                                </button>
                            </Col>
                        </Row>
                    </div>
                </Col>
            </div>
        );
    }
}
