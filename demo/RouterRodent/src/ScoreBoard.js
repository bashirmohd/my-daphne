import React from "react";
import { db } from "./fb";

export default class ScoreBoard extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            scores: []
        };
    }
    componentDidMount() {
        db.collection(`games/router-rodent/paths/${this.props.game}/scores`)
            .orderBy("time")
            .limit(10)
            .onSnapshot(result => {
                result.docs.forEach(doc => console.log(doc.data()));

                const scores = result.docs.map(doc => doc.data());
                console.log(scores);

                this.setState({ scores });
            });
    }

    render() {
        const { scores } = this.state;
        const rows = scores.map(score => (
            <tr style={{ fontSize: 30 }}>
                <td style={{ color: "white", width: 180 }}>{score.name}</td>
                <td style={{ color: "#4ec1e0", padding: 15 }}>
                    {score.time.toFixed(1)}
                </td>
            </tr>
        ));

        return <table>{rows}</table>;
    }
}
