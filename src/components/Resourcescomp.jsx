import React from "react";
import { Row, Col, Alert } from "react-bootstrap";
import { words } from "../helpers/lang";
import "../css/Resourcescard.css";
function formatNumber(x) {
  return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function Resourcescomp({ lang, resources }) {
  return (
    <div
    >
      <Row>
        <Col>
          <Resourcescard
            variant="info"
            title={words.ResourcesLegend.blue[lang]}
            value={formatNumber(resources.aqua)}
            lang={lang}
          />
        </Col>
        <Col>
          <Resourcescard
            variant="success"
            title={words.ResourcesLegend.green[lang]}
            value={formatNumber(resources.agriculture_land)}
            lang={lang}
          />
        </Col>
      </Row>
      <Row>
        <Col>
          <Resourcescard
            variant="warning"
            title={words.ResourcesLegend.yellow[lang]}
            value={formatNumber(resources["sand-rocks"])}
            lang={lang}
          />
        </Col>
        <Col>
          <Resourcescard
            variant="danger"
            title={words.ResourcesLegend.magenta[lang]}
            value={formatNumber(resources["urban-land"])}
            lang={lang}
          />
        </Col>
      </Row>
      <Resourcescard
        variant="primary"
        title={words.ResourcesLegend.purple[lang]}
        value={formatNumber(resources.road)}
        lang={lang}
      />
    </div>
  );
}

function Resourcescard({ lang , title, value, variant }) {
  return (
    <Alert
      variant={variant}
      style={{
        height: "4.5rem",
        width: "100%",
        textAlign: "center",
        justifyContent: "center",
        display: "flex",
      }}
    >
      <p>
        <strong>
          {lang === "en" ? <div>{title}: </div> : <div>:{title} </div>}

          {lang === "en" ? (
            <span>
              {value} {words.ResourcesLegend.metrics[lang]}
              <sup>2</sup>
            </span>
          ) : (
            <span>
              <span>
                {words.ResourcesLegend.metrics[lang]}
                <sup>٢</sup>{" "}
              </span>
              {value}
            </span>
          )}
        </strong>
      </p>
    </Alert>
  );
}

export default Resourcescomp;
