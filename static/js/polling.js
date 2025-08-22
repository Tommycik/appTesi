async function pollResult() {
                const res = await fetch("{url_for('get_result',job_id=job_id)}");
                const data = await res.json();
                if (data.status === "done") {
                    let resultDiv = document.getElementById("result-section");
                    if (data.output.startsWith("http")) {
                        resultDiv.innerHTML = `<img src="${data.output}" style='max-width: 500px;'/>`;
                    } else {
                        resultDiv.innerHTML = "<p>" + data.output + "</p>";
                    }
                } else {
                    setTimeout(pollResult, 2000);
                }
            }
            pollResult();