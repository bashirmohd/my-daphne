import React from "react";
import { GlobalHotKeys } from "react-hotkeys";
import { Row, Col, Container } from "react-bootstrap";

import Map from "./Map";
import Home from "./Home";
import Play from "./Play";

import icon from "./icon.png";
import "./App.css";

console.log("GlobalHotKeys", GlobalHotKeys);

class Game extends React.Component {
    constructor(props) {
        super(props);

        this.initialState = {
            currentGame: 0,
            current: "LBNL",
            path: ["LBNL", "SUNN"],
            screen: "startview",
            mapScreen: "playing",
            counter: 0,
            time: 0,
            cheat: false
        };

        this.state = this.initialState;
    }

    handleIncrement() {
        if (this.state.counter < 20) {
            this.setState({
                counter: this.state.counter + 1
            });
        }
    }
    handleDecrement() {
        if (this.state.counter > -20) {
            this.setState({
                counter: this.state.counter - 1
            });
        }
    }
    handleCheat() {
        this.setState({ cheat: !this.state.cheat });
    }
    beginGame(pathName, path) {
        this.setState({ screen: "mapview", pathName, path, current: path[0] });
    }

    playGame(source, dest, traffic) {
        this.setState({ screen: "gameview", source, dest, traffic });
    }

    savedGame() {
        this.setState({
            mapScreen: "saveCompleted"
        });
    }

    doneGame() {
        this.setState({
            time: 0,
            screen: "startview",
            mapScreen: "playing"
        });
    }

    levelComplete(t, next) {
        const { path } = this.state;

        let mapScreen = "playing";
        if (path[path.length - 1] === next) {
            // Game finished
            console.log("Finished game");
            mapScreen = "saving";
        }

        this.setState({
            current: next,
            screen: "mapview",
            mapScreen,
            time: this.state.time + t
        });
    }

    gameFinished() {
        this.setState({ screen: "endview" });
    }

    render() {
        let screen;
        if (this.state.screen === "startview") {
            screen = (
                <Home
                    onbeginGame={(name, path) => this.beginGame(name, path)}
                />
            );
        } else if (this.state.screen === "mapview") {
            screen = (
                <Map
                    current={this.state.current}
                    path={this.state.path}
                    pathName={this.state.pathName}
                    totalTime={this.state.time}
                    mapScreen={this.state.mapScreen}
                    onPlayGame={(source, dest, traffic) =>
                        this.playGame(source, dest, traffic)
                    }
                    onSavedGame={() => this.savedGame()}
                    onDoneGame={() => this.doneGame()}
                />
            );
        } else {
            const traffic = this.state.traffic / 100; // 0 to 100 Gbps
            const min = 0.04;
            const max = 0.3;
            let p = min + (max - min) * traffic;
            if (p > max) p = max;
            if (p < min) p = min;

            if (this.state.cheat) {
                p = 0.1;
            }
            screen = (
                <Play
                    source={this.state.source}
                    dest={this.state.dest}
                    pos={this.state.counter}
                    probability={p}
                    onLevelComplete={(t, next) => this.levelComplete(t, next)}
                />
            );
        }

        const globalKeyMap = {
            MOVE_LEFT: ["a", "A", "left"],
            MOVE_RIGHT: ["d", "D", "right"],
            CHEAT: ["p"]
        };

        const globalHandlers = {
            MOVE_LEFT: () => this.handleDecrement(),
            MOVE_RIGHT: () => this.handleIncrement(),
            CHEAT: () => this.handleCheat()
        };

        const hotkeys = (
            <GlobalHotKeys
                keyMap={globalKeyMap}
                handlers={globalHandlers}
                global
            />
        );

        return (
            <React.StrictMode>
                {hotkeys}
                <Container fluid={true} style={{ background: "#272c34" }}>
                    <Row style={{ backgroundColor: "#3a404c" }}>
                        <Col md={1}>
                            <a href="/">
                                <img
                                    width={75}
                                    src={icon}
                                    alt="icon"
                                    style={{ marginTop: 5 }}
                                />
                            </a>
                        </Col>

                        <Col md={11}>
                            <div>
                                <pre className="font">
                                    <h1
                                        style={{
                                            color: "#efefef",
                                            marginTop: 15
                                        }}
                                    >
                                        {`Router Rodent${
                                            this.state.cheat ? "*" : ""
                                        }`}
                                    </h1>
                                </pre>
                            </div>
                        </Col>
                    </Row>
                    <Row>
                        <Col>{screen}</Col>
                    </Row>
                </Container>
            </React.StrictMode>
        );
    }
}

export default Game;
