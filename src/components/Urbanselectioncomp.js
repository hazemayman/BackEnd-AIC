import React from "react";
import { Dropdown } from "react-bootstrap";

function Urbanselectioncomp() {
  const urbans = [
    { type: "Agriculture", color: "green " },
    { type: "Aqua", color: "indigo" },
    { type: "Sand", color: "#ffbf00" },
    { type: "Urban", color: "#808080" },
    { type: "Roads", color: "#404040" },
    { type: "Unknown", color: "white" },
  ];
  return (
    <Dropdown>
      <Dropdown.Toggle variant="Success" id="dropdown-basic">
        Urban
      </Dropdown.Toggle>
      <Dropdown.Menu style={{ overflowY: "scroll", maxHeight: "300px" }}>
        {urbans.map(({ type, color }) => {
          return (
            <Dropdown.Item
              style={{ justifyContent: "space-between" }}
              href={"#"}
            >
              {type}
            </Dropdown.Item>
          );
        })}
      </Dropdown.Menu>
    </Dropdown>
  );
}

export default Urbanselectioncomp;
