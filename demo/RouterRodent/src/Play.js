import React from "react";

import { Stage } from "@inlet/react-pixi";
import * as PIXI from "pixi.js";
import { PixiComponent } from "@inlet/react-pixi";
import { Row, Col } from "react-bootstrap";

//rodent
const Rodent = PixiComponent("rodent", {
    create: props => {
        return new PIXI.Graphics();
    },
    didMount: (instance, parent) => {},
    willUnmount: (instance, parent) => {},

    applyProps: (instance, oldProps, newProps) => {
        const { mousex, mousey } = newProps;
        instance.clear();

        instance.beginFill(0x6b757d);
        instance.lineStyle(2, 0x000, 5);
        instance.drawEllipse(
            250 + 200 + mousex * 20,
            mousey - 220 + 200,
            30,
            40
        );
        instance.endFill();

        instance.beginFill(0x6b757d);
        instance.lineStyle(3, 0x000, 5);
        instance.drawEllipse(
            225 + 200 + mousex * 20,
            mousey - 220 + 190,
            15,
            15
        );
        instance.endFill();

        instance.beginFill(0x6b757d);
        instance.lineStyle(3, 0x000, 5);
        instance.drawEllipse(
            275 + 200 + mousex * 20,
            mousey - 220 + 190,
            15,
            15
        );
        instance.endFill();

        instance.beginFill(0x6b757d);
        instance.lineStyle(2, 0x6b757d, 0);
        instance.drawEllipse(
            265 + 200 + mousex * 20,
            mousey - 220 + 186,
            12,
            14
        );
        instance.endFill();

        instance.beginFill(0x6b757d);
        instance.lineStyle(2, 0x6b757d, 0);
        instance.drawEllipse(
            235 + 200 + mousex * 20,
            mousey - 220 + 186,
            12,
            14
        );
        instance.endFill();

        instance.beginFill(0x000);
        instance.drawEllipse(259 + 200 + mousex * 20, mousey - 220 + 177, 3, 4);
        instance.endFill();

        instance.beginFill(0x000);
        instance.drawEllipse(241 + 200 + mousex * 20, mousey - 220 + 177, 3, 4);
        instance.endFill();

        instance.beginFill(0x0000, 0);
        instance.lineStyle(4, 0x6b757d, 1);
        instance.drawEllipse(255 + 200 + mousex * 20, mousey - 220 + 243, 5, 9);
        instance.endFill();

        instance.beginFill(0x0000);
        instance.lineStyle(1, 0x6b757d, 0);
        instance.drawEllipse(
            259 + 200 + mousex * 20,
            mousey - 220 + 249,
            5,
            12
        );
        instance.endFill();
    }
});

//obstacles
const Square = PixiComponent("square", {
    create: props => {
        return new PIXI.Graphics();
    },
    didMount: (instance, parent) => {},
    willUnmount: (instance, parent) => {},

    applyProps: (instance, oldProps, newProps) => {
        const size = 20;
        const { posx, posy } = newProps;
        const fillColor = 0x61656b;
        instance.clear();
        instance.beginFill(fillColor);
        instance.lineStyle(1, 0xffffff, 1);
        instance.moveTo(posx, posy);
        instance.lineTo(posx, posy + size);
        instance.lineTo(posx - size, posy + size);
        instance.lineTo(posx - size, posy);
        instance.lineTo(posx, posy);
        instance.endFill();
    }
});

function doesIntersect(a, b) {
    return (
        Math.abs(a.x - b.x) * 2 < a.width + b.width &&
        Math.abs(a.y - b.y) * 2 < a.height + b.height
    );
}

export default class PlayView extends React.Component {
    constructor(props) {
        super(props);

        this.gridSizeX = 10;
        this.gridSizeY = 50;

        const generatedGrid = this.createGrid(
            this.gridSizeX,
            this.gridSizeY,
            this.props.probability
        );

        this.state = {
            d: 0,
            beginTime: new Date(),
            currentTime: new Date(),
            progress: 0,
            grid: generatedGrid,
            levelState: "play",
            mousePosition: 555
        };

        this.updateAnimationState = this.updateAnimationState.bind(this);
    }

    componentDidMount() {
        this.rAF = requestAnimationFrame(this.updateAnimationState);
    }

    updateAnimationState() {
        const prevState = this.state;

        const { levelState } = this.state;

        const t = new Date();

        if (levelState === "play") {
            let speed = 0.4;

            let dmin = 1000;
            const mousex = 425 + this.props.pos * 20 + 30;
            const mousey = 555 + 25 / 2 - 25;

            for (let i = 0; i < this.gridSizeX; i++) {
                for (let j = 0; j < this.gridSizeY; j++) {
                    const hasTrap = this.state.grid[i][j] !== null;
                    if (hasTrap) {
                        const { xpos, ypos } = this.state.grid[i][j];

                        const posx = xpos;
                        const posy = ypos + prevState.progress * 10;

                        const trapx = posx - 10;
                        const trapy = posy + 10;

                        const dx = trapx - mousex;
                        const dy = trapy - mousey;

                        const d = Math.sqrt(dx * dx + dy * dy);

                        if (d < dmin) {
                            dmin = d;
                        }
                    }
                }
            }

            const minSpeed = 0.5;
            const maxSpeed = 0.9;

            speed = ((maxSpeed - minSpeed) * (dmin - 50)) / 150 + minSpeed;

            if (speed < minSpeed) {
                speed = minSpeed;
            }
            if (speed > maxSpeed) {
                speed = maxSpeed;
            }

            let newProgress = prevState.progress + speed;

            let transitionToFinishing = false;
            if (newProgress > 560) {
                transitionToFinishing = true;
            }

            // Detect collisions
            const collides = dmin < 50;
            if (collides) {
                newProgress = 0;
            }

            this.setState({
                d: dmin,
                currentTime: t,
                progress: newProgress,
                levelState: transitionToFinishing ? "finishing" : "play"
            });

            this.rAF = requestAnimationFrame(this.updateAnimationState);
        } else if (levelState === "finishing") {
            let transitionToComplete = false;
            if (prevState.mousePosition < 0) {
                transitionToComplete = true;
            }

            this.setState({
                currentTime: t,
                mousePosition: prevState.mousePosition - 4,
                levelState: transitionToComplete ? "complete" : "finishing"
            });

            this.rAF = requestAnimationFrame(this.updateAnimationState);
        }
    }
    componentWillUnmount() {
        cancelAnimationFrame(this.rAF);
    }

    handleIncrement() {
        this.setState({
            progress: this.state.progress + 1
        });
    }
    handleDecrement() {
        this.setState({
            progress: this.state.progress - 1
        });
    }

    createGrid(width, height, p) {
        const grid = [];
        for (let i = 0; i < width; i++) {
            grid[i] = [];
            for (let j = 0; j < height; j++) {
                const r = Math.random();
                const offsetScale = 20;
                const xoffset = (Math.random() - 0.5) * 2 * offsetScale;
                const yoffset = (Math.random() - 0.5) * 2 * offsetScale;
                if (r < p) {
                    grid[i][j] = {
                        xpos: i * 100 + xoffset,
                        ypos: j * -100 + yoffset
                    };
                } else {
                    grid[i][j] = null;
                }
            }
        }
        return grid;
    }

    render() {
        const traps = [];
        for (let i = 0; i < this.gridSizeX; i++) {
            for (let j = 0; j < this.gridSizeY; j++) {
                if (this.state.grid[i][j] !== null) {
                    const { xpos, ypos } = this.state.grid[i][j];
                    traps.push(
                        <Square
                            key={`trap-${i}-${j}`}
                            posx={xpos}
                            posy={ypos + this.state.progress * 10}
                        />
                    );
                }
            }
        }

        // Time the game has been running
        const delta =
            (this.state.currentTime.getTime() -
                this.state.beginTime.getTime()) /
            1000;

        // Which screen to display
        let screen;
        if (
            this.state.levelState === "play" ||
            this.state.levelState === "finishing"
        ) {
            // Playing the game
            screen = (
                <div ref={this.stage}>
                    <Stage
                        width={900}
                        height={600}
                        color={"#CCC"}
                        options={{ antialias: true }}
                    >
                        <Rodent
                            mousex={this.props.pos}
                            mousey={this.state.mousePosition}
                        />
                        {traps}
                    </Stage>
                </div>
            );
        } else if (this.state.levelState === "complete") {
            // Display at the end of the game
            screen = (
                <Row>
                    <Col md={12}>
                        <div>
                            <pre>
                                <h2
                                    className="font"
                                    style={{
                                        color: "#efefef",
                                        marginTop: 15
                                    }}
                                >
                                    Level complete
                                </h2>
                            </pre>
                        </div>
                    </Col>

                    <Col md={12}>
                        <span>
                            <button
                                type="button"
                                className="font btn btn-secondary"
                                onClick={() =>
                                    this.props.onLevelComplete(
                                        delta,
                                        this.props.dest
                                    )
                                }
                                style={{
                                    marginTop: 7
                                }}
                            >
                                <pre>
                                    <h3 style={{ marginBottom: 1 }}>
                                        Continue
                                    </h3>
                                </pre>
                            </button>
                        </span>
                    </Col>
                </Row>
            );
        }

        return (
            <Col>
                <Row>
                    <Col md={{ span: 2, offset: 5 }}>
                        <pre style={{ color: "white" }}>
                            {`${this.props.source} to ${this.props.dest}`}
                        </pre>
                    </Col>
                </Row>
                <Row>
                    <Col md={{ span: 2, offset: 6 }}>
                        <pre style={{ color: "white" }}>{`${delta.toFixed(
                            1
                        )}s`}</pre>
                    </Col>
                </Row>
                <Row>
                    <Col md={{ span: 6, offset: 2 }}>
                        {/* <div style={{ display: "flex", flexDirection: "row" }}>
                        <div style={{ flex: 0 }} />
                        <div style={{ flex: 1 }}> */}
                        <center>
                            <Col>{screen}</Col>
                        </center>
                        {/* </div>
                        <div style={{ flex: 0 }} /> */}
                        {/* </div> */}
                    </Col>
                </Row>
            </Col>
        );
    }
}
