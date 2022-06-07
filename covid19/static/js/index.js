$(document).ready(function(){
    $("#btn_search_user_query").click(issue_query);

    let $input_box = $("#input_user_query");
    $input_box.keypress(function (e) {
        let key = e.which;
        if(key === 13) { // the enter key code
            issue_query();
        }
    });
    function issue_query() {
        let query_text = $input_box.val();
        window.location.href = "/search/?q=" + encodeURIComponent(query_text);
    }

});