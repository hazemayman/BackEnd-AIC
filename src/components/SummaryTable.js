import React from "react";
import "../css/Table.css";
function Table({ govs }) {
  return (
    <div className="table">
      <tr>
        <th>City</th>
        <th>Agriculture</th>
        <th>Aqua</th>
        <th>Sand</th>
        <th>Urban</th>
        <th>Roads</th>
        <th>No data</th>
      </tr>

      {govs.map(({ gov, agri }) => (
        <tr>
          <td>{gov}</td>
          <td>
            <strong>12%</strong>
          </td>
        </tr>
      ))}
    </div>
  );
}

export default Table;
