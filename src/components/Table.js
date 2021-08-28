import React from "react";

function Table() {
  return (
    <div className="table">
      <thead>
        <tr>
          <th>City</th>
          <th>Agriculture</th>
          <th>Aqua</th>
          <th>Sand</th>
          <th>Urban</th>
          <th>Roads</th>
          <th>No data</th>
        </tr>
      </thead>
      <tbody>
        {govs.map(({ gov, agri }) => (
          <tr>
            <td>{gov}</td>
            <td>
              <strong>12%</strong>
            </td>
          </tr>
        ))}
      </tbody>
    </div>
  );
}

export default Table;
