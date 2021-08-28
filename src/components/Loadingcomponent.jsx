import React from "react";
import { Spinner } from "react-bootstrap";
function Loadingcomponent() {
  return (
    <div
      style={{
        alignItems: "center",
        display: "flex",
        justifyContent: "center",
        width: "100%",
        height: "100%",
        position: "absolute",
      }}
    >
      <Spinner animation="border" variant="danger" size="lg" />
    </div>
  );
}

export default Loadingcomponent;
