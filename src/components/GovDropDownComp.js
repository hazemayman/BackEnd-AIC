import React, { useState, useEffect } from "react";
import { Dropdown } from "react-bootstrap";
import { words } from "../helpers/lang";
import { urls } from "../helpers/urls";

function GovDropDownComp({ lang, govs, fetchSelection }) {
  // add the real

  return (
    <Dropdown>
      <Dropdown.Toggle variant="dark" id="dropdown-basic">
        {words.GovDropDownTitle[lang]}
      </Dropdown.Toggle>
      <Dropdown.Menu style={{ overflowY: "scroll", maxHeight: "300px" }}>
        <Dropdown.Item href={"/"}>All Governorates</Dropdown.Item>
        {govs.map((gov) => {
          return (
            <Dropdown.Item
              onClick={() => {
                fetchSelection(gov);
              }}
            >
              {gov.replace(/-/g, " ")}
            </Dropdown.Item>
          );
        })}
      </Dropdown.Menu>
    </Dropdown>
  );
}

export default GovDropDownComp;
