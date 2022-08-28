import React from "react";
import { Row, Col } from "react-bootstrap";
import { TrafficMap } from "react-network-diagrams";

import ScoreSaver from "./ScoreSaver";
import ScoreBoard from "./ScoreBoard";

import data from "./data.json";

const nodeSizeMap = {
    node: 9,
    completed: 9,
    remaining: 9
};

export default class Map extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            edgeMap: null
        };
    }
    async componentDidMount() {
        try {
            const response = await fetch("traffic");
            const data = await response.json();
            const edges = data.data.mapTopology.edges;
            const edgeMap = {};
            edges.forEach(edge => {
                const timeseries = JSON.parse(edge.netbeamTraffic);
                const trafficAZ = timeseries.points[0][1];
                const trafficZA = timeseries.points[0][2];
                edgeMap[edge.name] = trafficAZ;
                const [src, dest] = edge.name.split("--");
                edgeMap[`${dest}--${src}`] = trafficZA;
            });
            this.setState({ edgeMap });
        } catch (err) {
            console.log("Edge fetching error", err);
        }
    }
    render() {
        const current = this.props.current;
        const path = this.props.path;

        const completed = [];
        const remaining = [];
        let beforeCurrent = true;
        path.forEach(p => {
            if (p === current) {
                beforeCurrent = false;
                completed.push(p);
                remaining.push(p);
            } else {
                if (beforeCurrent === true) {
                    completed.push(p);
                } else {
                    remaining.push(p);
                }
            }
        });

        const nodes = data.data.mapTopology.nodes.map(n => {
            const { name, x, y, ...other } = n;
            let type = "node";
            if (remaining.includes(name)) {
                type = "remaining";
            } else if (completed.includes(name)) {
                type = "completed";
            }
            return {
                name,
                type,
                x: x,
                y: y * 1.2,
                ...other
            };
        });

        const edges = data.data.mapTopology.edges.map(e => {
            const { name } = e;
            const parts = name.split("--");
            const source = parts[0];
            const target = parts[1];
            return {
                source,
                target
            };
        });

        const paths = [
            {
                steps: completed,
                name: "completed"
            },
            {
                steps: remaining,
                name: "remaining"
            }
        ];

        const topology = {
            nodes,
            edges,
            paths
        };

        const bounds = {
            x1: -5,
            y1: 5,
            x2: 240,
            y2: 140
        };

        const edgeThicknessMap = {};

        const nodeSyleMap = {
            node: {
                normal: {
                    fill: "#CBCBCB",
                    stroke: "#BEBEBE",
                    cursor: "pointer"
                },
                selected: {
                    fill: "#37B6D3",
                    stroke: "rgba(55, 182, 211, 0.22)",
                    strokeWidth: 10,
                    cursor: "pointer"
                },
                muted: {
                    fill: "#CBCBCB",
                    stroke: "#BEBEBE",
                    opacity: 0.6,
                    cursor: "pointer"
                }
            },
            label: {
                normal: {
                    fill: "#696969",
                    stroke: "none",
                    fontSize: 9,
                    opacity: 0
                },
                selected: {
                    fill: "#333",
                    stroke: "none",
                    fontSize: 11,
                    opacity: 0
                },
                muted: {
                    fill: "#696969",
                    stroke: "none",
                    fontSize: 8,
                    opacity: 0
                }
            }
        };

        const nodeHighlightedSyleMap = {
            node: {
                normal: {
                    fill: "#CBCBCB",
                    stroke: "#4ec1e0",
                    strokeWidth: 4,
                    cursor: "pointer"
                },
                selected: {
                    fill: "#37B6D3",
                    stroke: "rgba(55, 182, 211, 0.22)",
                    strokeWidth: 10,
                    cursor: "pointer"
                },
                muted: {
                    fill: "#CBCBCB",
                    stroke: "#BEBEBE",
                    opacity: 0.6,
                    cursor: "pointer"
                }
            },
            label: {
                normal: {
                    fill: "#696969",
                    stroke: "none",
                    fontSize: 9,
                    opacity: 0
                },
                selected: {
                    fill: "#333",
                    stroke: "none",
                    fontSize: 11,
                    opacity: 0
                },
                muted: {
                    fill: "#696969",
                    stroke: "none",
                    fontSize: 8,
                    opacity: 0
                }
            }
        };

        const nodeMutedSyleMap = {
            node: {
                normal: {
                    fill: "#CBCBCB",
                    stroke: "#4ec1e0",
                    strokeWidth: 2,
                    cursor: "pointer"
                },
                selected: {
                    fill: "#37B6D3",
                    stroke: "rgba(55, 182, 211, 0.22)",
                    strokeWidth: 10,
                    cursor: "pointer"
                },
                muted: {
                    fill: "#CBCBCB",
                    stroke: "#BEBEBE",
                    opacity: 0.6,
                    cursor: "pointer"
                }
            },
            label: {
                normal: {
                    fill: "#696969",
                    stroke: "none",
                    fontSize: 9,
                    opacity: 0
                },
                selected: {
                    fill: "#333",
                    stroke: "none",
                    fontSize: 11,
                    opacity: 0
                },
                muted: {
                    fill: "#696969",
                    stroke: "none",
                    fontSize: 8,
                    opacity: 0
                }
            }
        };

        // Mapping of node type to style
        const stylesMap = {
            node: nodeSyleMap,
            completed: nodeMutedSyleMap,
            remaining: nodeHighlightedSyleMap
        };

        const pathColorMap = {
            completed: "#4ec1e04a",
            remaining: "#4ec1e0"
        };

        const pathWidthMap = {
            completed: 5,
            remaining: 5
        };

        const showPaths = ["completed", "remaining"];

        const source = remaining[0];
        const dest = remaining[1];

        console.log("Map rendering, screen=", this.props.mapScreen);

        // Get the edge traffic for this source and dest, if we have the edgeMap yet
        const edge = `${source}--${dest}`;
        let traffic = 0;
        if (this.state.edgeMap) {
            traffic = this.state.edgeMap[edge] / 1000000000;
        }

        const trafficLabel = traffic > 0 ? `${parseInt(traffic, 10)} Gbps` : "";

        const playButton =
            this.props.mapScreen === "saving" ||
            this.props.mapScreen === "saveCompleted" ? (
                <div />
            ) : (
                <button
                    onClick={() => this.props.onPlayGame(source, dest, traffic)}
                    type="button"
                    className="font btn btn-secondary"
                >
                    <pre
                        style={{
                            marginBottom: 1,
                            marginTop: 7
                        }}
                    >
                        <h3>Play</h3>
                    </pre>
                </button>
            );

        const statusRow =
            this.props.mapScreen === "saving" ||
            this.props.mapScreen === "saveCompleted" ? (
                <Row>
                    <Col
                        sm={5}
                        className="font"
                        style={{
                            color: "#efefef",
                            marginTop: 15,
                            marginLeft: 50
                        }}
                    >
                        Game completed!
                    </Col>
                    <Col
                        sm={2}
                        className="font"
                        style={{
                            color: "#efefef",
                            marginTop: 15,
                            marginBottom: 15
                        }}
                    >
                        {playButton}
                    </Col>
                    <Col
                        sm={3}
                        className="font"
                        style={{
                            color: "#efefef",
                            marginTop: 15
                        }}
                    >
                        <Row>Final time:</Row>
                        <Row>{this.props.totalTime.toFixed(1)}</Row>
                    </Col>
                </Row>
            ) : (
                <Row>
                    <Col
                        sm={2}
                        className="font"
                        style={{
                            color: "#efefef",
                            marginTop: 15,
                            marginLeft: 50
                        }}
                    >
                        <Row>Source:</Row>
                        <Row>{source}</Row>
                    </Col>
                    <Col
                        sm={2}
                        className="font"
                        style={{
                            color: "#efefef",
                            marginTop: 15
                        }}
                    >
                        <Row>Destination:</Row>
                        <Row>{dest}</Row>
                    </Col>
                    <Col
                        sm={2}
                        className="font"
                        style={{
                            color: "#efefef",
                            marginTop: 15,
                            marginBottom: 15
                        }}
                    >
                        {playButton}
                    </Col>
                    <Col
                        sm={2}
                        className="font"
                        style={{
                            color: "#efefef",
                            marginTop: 15
                        }}
                    >
                        <Row>Time:</Row>
                        <Row>{this.props.totalTime.toFixed(1)}</Row>
                    </Col>
                    <Col
                        sm={2}
                        className="font"
                        style={{
                            color: "#efefef",
                            marginTop: 15
                        }}
                    >
                        <Row>Traffic:</Row>
                        <Row>{trafficLabel}</Row>
                    </Col>
                </Row>
            );

        let mainRow;

        if (this.props.mapScreen === "saving") {
            mainRow = (
                <Row>
                    <Col md={12} style={{ background: "#282c34" }}>
                        <div
                            style={{
                                marginTop: 5,
                                background: "#1a1f23",
                                height: 600
                            }}
                        >
                            <ScoreSaver
                                score={this.props.totalTime}
                                game={this.props.pathName}
                                onSaved={() => this.props.onSavedGame()}
                            />
                        </div>
                    </Col>
                </Row>
            );
        } else if (this.props.mapScreen === "saveCompleted") {
            mainRow = (
                <Row>
                    <Col md={12} style={{ background: "#282c34" }}>
                        <div
                            style={{
                                marginTop: 5,
                                background: "#1a1f23",
                                height: 600
                            }}
                        >
                            <pre>
                                <h1 style={{ color: "white", marginTop: 80 }}>
                                    Saved!
                                </h1>
                            </pre>
                            <button
                                type="button"
                                className="font btn btn-secondary"
                                onClick={() => this.props.onDoneGame()}
                                style={{
                                    marginTop: 20,
                                    marginBottom: 0
                                }}
                            >
                                <pre>
                                    <h3
                                        style={{
                                            marginTop: 10,
                                            marginBottom: 10,
                                            marginLeft: 20,
                                            marginRight: 20
                                        }}
                                    >
                                        CONTINUE
                                    </h3>
                                </pre>
                            </button>
                        </div>
                    </Col>
                </Row>
            );
        } else if (this.props.mapScreen === "playing") {
            mainRow = (
                <Row>
                    <Col md={12} style={{ background: "#282c34" }}>
                        <div style={{ marginTop: 5 }}>
                            <TrafficMap
                                style={{ background: "#1b1f24" }}
                                topology={topology}
                                height={300}
                                bounds={bounds}
                                edgeDrawingMethod="simple"
                                nodeSizeMap={nodeSizeMap}
                                edgeThicknessMap={edgeThicknessMap}
                                stylesMap={stylesMap}
                                pathColorMap={pathColorMap}
                                pathWidthMap={pathWidthMap}
                                showPaths={showPaths}
                            />
                        </div>
                    </Col>
                </Row>
            );
        }

        return (
            <div className="App">
                <Col style={{ backgroundColor: "#3a404c" }}>
                    <Row style={{ backgroundColor: "#282c34" }}>
                        <Col sm={9} style={{ background: "#282c34" }}>
                            {statusRow}
                            {mainRow}
                        </Col>
                        <Col
                            sm={3}
                            style={{ background: "#1b1f24", marginTop: 10 }}
                        >
                            <pre className="font">
                                <h2
                                    style={{
                                        color: "#FFF",
                                        marginTop: 10,
                                        fontSize: 40
                                    }}
                                >
                                    Scoreboard
                                </h2>
                                <ScoreBoard game={this.props.pathName} />
                            </pre>
                        </Col>
                    </Row>
                    <Row style={{ height: 30, backgroundColor: "#282c34" }} />
                </Col>
            </div>
        );
    }
}
