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
        let quote = $input_box.val();
        window.location.href = "/quote/?q=" + encodeURIComponent(quote);
    }

    // Keep track of number of articles to display based on screen size.
    let width;
    let numArticles;
    setInterval(() => onResize(), 300);

    function onResize() {
        width = $(window).width();
        let oldNum = numArticles;
        if (width >= 748) {
            numArticles = 2;
        } else {
            numArticles = 1;
        }
        if (width < 576) {
            $(".arrow").hide();
        } else {
            $(".arrow").show();
        }
        if (oldNum !== numArticles) {
            redraw(numArticles);
        }
    }

    function redraw(numArticles) {
        $(".previous").each(function() {
            let $id = this.id.split("-");
            let $source = $id[0];
            let $position = $id[1];
            if (parseInt($position) === 0) {
                $(this).css('visibility', 'hidden');
            }
            if ($(".article." + $source + "." + (parseInt($position) + numArticles).toString()).length === 0) {
                $("#" + $source + "-" + $position + ".next").css('visibility', 'hidden');
            }
            $(".article." + $source).hide();
            for (let i = 0; i < numArticles; i++) {
                $(".article." + $source + "." + (parseInt($position) + i).toString()).show();
            }
        });
    }

    $(".next").click(function() {
        let $id = this.id.split("-");
        let $source = $id[0];
        let $oldPosition = $id[1];
        let $newPosition = (parseInt($oldPosition)+numArticles).toString();
        $(this).attr('id', $source + "-" + $newPosition);
        $("#" + $source + "-" + $oldPosition).attr('id', $source + "-" + $newPosition);
        for (let i = 0; i < numArticles; i++) {
            $(".article." + $source + "." + (parseInt($newPosition) + i).toString()).show();
            $(".article." + $source + "." + (parseInt($oldPosition) + i).toString()).hide();
        }
        if ($(".article." + $source + "." + (parseInt($newPosition)+numArticles).toString()).length === 0) {
            $(this).css('visibility','hidden');
        }
        $("#" + $source + "-" + $newPosition + ".previous").css('visibility','visible');
    });

    $(".previous").click(function() {
        let $id = this.id.split("-");
        let $source = $id[0];
        let $oldPosition = $id[1];
        let $newPosition = (Math.max(0, parseInt($oldPosition)-numArticles)).toString();
        $(this).attr('id', $source + "-" + $newPosition);
        $("#" + $source + "-" + $oldPosition).attr('id', $source + "-" + $newPosition);
        for (let i = 0; i < numArticles; i++) {
            $(".article." + $source + "." + (parseInt($newPosition) + i).toString()).show();
            $(".article." + $source + "." + (parseInt($oldPosition) + i).toString()).hide();
        }
        if (parseInt($newPosition) === 0) {
            $(this).css('visibility','hidden');
        }
        $("#" + $source + "-" + $newPosition + ".next").css('visibility','visible');
    });
});