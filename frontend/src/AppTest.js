import React from 'react';

function AppTest() {
  return (
    <div style={{
      padding: '50px',
      backgroundColor: 'white',
      color: 'black',
      minHeight: '100vh'
    }}>
      <h1 style={{ 
        fontSize: '32px', 
        fontWeight: 'bold',
        marginBottom: '20px',
        color: '#11052C'
      }}>
        CourtCheck - Tennis Match Analysis
      </h1>
      
      <div style={{
        padding: '30px',
        backgroundColor: '#f0f0f0',
        borderRadius: '10px',
        marginTop: '20px'
      }}>
        <h2 style={{ marginBottom: '15px', fontSize: '24px' }}>
          React is Working!
        </h2>
        <p style={{ fontSize: '16px', marginBottom: '10px' }}>
          If you can see this text, React is rendering correctly.
        </p>
        <p style={{ fontSize: '16px' }}>
          The issue might be with Tailwind CSS or the upload component.
        </p>
      </div>

      <div style={{
        marginTop: '30px',
        padding: '20px',
        backgroundColor: '#e3f2fd',
        borderRadius: '8px'
      }}>
        <h3 style={{ marginBottom: '10px', fontSize: '20px' }}>
          Next Steps:
        </h3>
        <ol style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>If you see this, React is working</li>
          <li>Check browser console (F12) for any errors</li>
          <li>Let me know what you see</li>
        </ol>
      </div>

      <div style={{ marginTop: '20px', fontSize: '14px', color: '#666' }}>
        <strong>Current Status:</strong>
        <ul style={{ paddingLeft: '20px', marginTop: '10px' }}>
          <li>Backend: Running on port 8000 ✅</li>
          <li>Frontend: Running on port 3000 ✅</li>
          <li>Webpack: Compiled successfully ✅</li>
          <li>React: {React.version} ✅</li>
        </ul>
      </div>
    </div>
  );
}

export default AppTest;
