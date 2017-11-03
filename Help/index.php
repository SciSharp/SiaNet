<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>

<head>
<title>Sia.DNN - Table of Content</title>
<link rel="stylesheet" href="TOC.css">
<link rel="shortcut icon" href="favicon.ico"/>
<script type="text/javascript" src="TOC.js"></script>
</head>

<body onload="javascript: Initialize('.php');" onresize="javascript: ResizeTree();">
<form id="IndexForm" runat="server">

<div id="TOCDiv" class="TOCDiv">

<div id="divSearchOpts" class="SearchOpts" style="height: 90px; display: none;">
	<img class="TOCLink" onclick="javascript: ShowHideSearch(false);"
		src="Close.png" height="24" width="24" alt="Hide Search" title="Hide Search" style="float: right;"/>
	<span class="Title">Keyword(s) to search:</span>
	<input id="txtSearchText" type="text" style="width: 100%;" 
		onkeypress="javascript: return OnSearchTextKeyPress(event);" /><br />
	<input id="chkSortByTitle" type="checkbox" /><label for="chkSortByTitle">&nbsp;Sort results by title</label>
	<input type="button" value="Search" onclick="javascript: return PerformSearch();" style="float: right;" />
</div>

<div id="divIndexOpts" class="IndexOpts" style="height: 28px; display: none;">
	<img class="TOCLink" onclick="javascript: ShowHideIndex(false);"
		src="Close.png" height="24" width="24" alt="Hide Keyword Index" title="Hide Keyword Index" style="float: right;"/>
	<span class="Title">Keyword Index</span>
</div>

<div id="divNavOpts" class="NavOpts" style="height: 28px;">
    <img class="TOCLink" onclick="javascript: SyncTOC();" src="SyncTOC.png"
        height="24" width="24" alt="Sync to TOC" title="Sync to TOC" />
    <img class="TOCLink" onclick="javascript: ExpandOrCollapseAll(true);"
        src="ExpandAll.png" height="24" width="24" alt="Expand all" title="Expand all" />
    <img class="TOCLink" onclick="javascript: ExpandOrCollapseAll(false);"
        src="CollapseAll.png" height="24" width="24" alt="Collapse all" title="Collapse all" />
    <img class="TOCLink" onclick="javascript: ShowHideIndex(true);"
        src="Index.png" height="24" width="24" alt="Keyword Index" title="Keyword Index" />
    <img class="TOCLink" onclick="javascript: ShowHideSearch(true);"
        src="Search.png" height="24" width="24" alt="Search" title="Search" />
    <img class="TOCLink" onclick="javascript: ShowDirectLink();"
        src="Link.png" height="24" width="24" alt="Get direct link to the displayed topic" title="Get direct link to the displayed topic" />
    <a href="#" onclick="javascript: TopicContent.history.go(-1); return false;">
		<img class="TOCLink" style="float: right;" 
        src="Back.png" height="24" width="24" alt="Previous page" title="Previous page" /></a>
</div>

<div class="Tree" id="divSearchResults" style="display: none;"
    onselectstart="javascript: return false;">
</div>

<div class="Tree" id="divIndexResults" style="display: none;"
    onselectstart="javascript: return false;">
</div>

<div class="Tree" id="divTree" onselectstart="javascript: return false;">
<?
$toc = new DOMDocument();
$toc->load('WebTOC.xml');
$xpath = new DOMXPath($toc);
$nodes = $xpath->query("/HelpTOC/*");
foreach($nodes as $node)
{
    $id = $node->getAttribute("Id");
    $url = $node->getAttribute("Url");
    $title = $node->getAttribute("Title");
    if (empty($url))
    {
        $url = "#";
        $target = "";
    }
    else
    {
        $target = " target=\"TopicContent\"";
    }

    if ($node->hasChildNodes())
    {
?>
        <div class="TreeNode">
            <img class="TreeNodeImg" onclick="javascript: Toggle(this);" src="Collapsed.png"/>
            <a class="UnselectedNode" onclick="javascript: Expand(this);" href="<?= $url ?>"<?= $target ?>><?= $title ?></a>
            <div id="<?= $id ?>" class="Hidden"></div>
        </div>
<?
    }
    else
    {
?>
        <div class="TreeItem">
            <img src="Item.gif"/>
            <a class="UnselectedNode" onclick="javascript: SelectNode(this);" href="<?= $url ?>"<?= $target ?>><?= $title ?></a>
        </div>
<?
    }
}
?>
</div>

</div>

<div id="TOCSizer" class="TOCSizer" onmousedown="OnMouseDown(event)" onselectstart="javascript: return false;"></div>

<iframe id="TopicContent" name="TopicContent" class="TopicContent" src="html/N-SiaNet.htm">
This page uses an IFRAME but your browser does not support it.
</iframe>

</form>

</body>

</html>