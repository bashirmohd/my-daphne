import React from "react";
import { db } from "./fb";

export default class ScoreSaver extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            overLetter: null,
            name: []
        };
    }
    handleLetterClick(letter) {
        console.log(letter);
        const name = this.state.name;
        if (name.length < 3) {
            name.push(letter);
            this.setState({ name });
        }
    }
    handleLetterBackspace() {
        const name = this.state.name;
        name.pop();
        this.setState({ name });
    }
    async saveScore() {
        const score = this.props.score;
        const name = this.state.name.join("").toUpperCase();
        const game = this.props.game;
        try {
            const result = await db
                .collection(`games/router-rodent/paths/${game}/scores`)
                .add({
                    name,
                    time: score
                });
            console.log("Firebase result", result);
            // Need to do something here to show a saved screen
            this.props.onSaved();
        } catch (err) {
            console.log("Firebase error", err);
        }
    }
    render() {
        console.log(this.props.path);
        const letters = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "_"
        ];

        const letterSelections = letters.map(letter => {
            const style = {
                fontSize: 36,
                padding: 5,
                cursor: "pointer",
                color: this.state.overLetter === letter ? "#4ec1e0" : "white"
            };
            return (
                <span
                    style={style}
                    onMouseEnter={() => this.setState({ overLetter: letter })}
                    onMouseOut={() => this.setState({ overLetter: null })}
                    onClick={() => this.handleLetterClick(letter)}
                >
                    {letter.toUpperCase()}
                </span>
            );
        });

        letterSelections.push(
            <span
                style={{ fontSize: 40, color: "white" }}
                onClick={() => this.handleLetterBackspace()}
            >
                ⌫
            </span>
        );

        const initials = ["⚊", "⚊", "⚊"];
        this.state.name.forEach((letter, index) => {
            initials[index] = letter.toUpperCase();
        });

        return (
            <div style={{ paddingTop: 100 }}>
                <div style={{ userSelect: "none" }}>
                    <pre
                        style={{
                            height: 50,
                            color: "#4ec1e0",
                            fontSize: 32
                        }}
                    >
                        Enter your initials
                    </pre>
                </div>
                <div style={{ userSelect: "none" }}>
                    <pre
                        style={{
                            height: 50,
                            color: "white",
                            fontSize: 32
                        }}
                    >
                        {initials.join("")}
                    </pre>
                </div>
                <div>
                    <pre style={{ height: 60 }}>{letterSelections}</pre>
                </div>
                <div>
                    <button
                        type="button"
                        className="font btn btn-secondary"
                        onClick={() => this.saveScore()}
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
                                SAVE
                            </h3>
                        </pre>
                    </button>
                </div>
            </div>
        );
    }
}
