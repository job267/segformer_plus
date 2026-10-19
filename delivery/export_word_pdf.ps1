param([Parameter(Mandatory=$true)][string]$InputDocx, [Parameter(Mandatory=$true)][string]$OutputPdf)
$ErrorActionPreference='Stop'
$taskDocPath=(Resolve-Path -LiteralPath $InputDocx).Path
$taskPdfPath=[System.IO.Path]::GetFullPath($OutputPdf)
$taskWord=$null
$taskDocument=$null
$taskOwnInstance=$false
try {
    $taskWord=New-Object -ComObject Word.Application
    if ($taskWord.Documents.Count -ne 0) { throw 'A new empty conversion instance was required; existing documents were not touched.' }
    $taskOwnInstance=$true
    $taskWord.Visible=$false
    $taskWord.DisplayAlerts=0
    $taskWord.AutomationSecurity=3
    $taskDocument=$taskWord.Documents.Open($taskDocPath,$false,$false,$false)
    $taskDocument.Fields.Update() | Out-Null
    foreach ($taskToc in $taskDocument.TablesOfContents) {
        $taskToc.Update()
        $taskToc.Range.Font.Size=10.5
        $taskToc.Range.Font.Bold=0
        $taskToc.Range.Font.Italic=0
        $taskToc.Range.Font.Color=0
        $taskToc.Range.Font.NameFarEast='宋体'
        $taskToc.Range.ParagraphFormat.SpaceBefore=0
        $taskToc.Range.ParagraphFormat.SpaceAfter=0
        $taskToc.Range.ParagraphFormat.LineSpacingRule=4
        $taskToc.Range.ParagraphFormat.LineSpacing=14
    }
    $taskDocument.Repaginate()
    foreach ($taskToc in $taskDocument.TablesOfContents) { $taskToc.UpdatePageNumbers() }
    $taskDocument.Save()
    $taskDocument.ExportAsFixedFormat($taskPdfPath,17)
    @{engine='Microsoft Word';version=$taskWord.Version;pages=$taskDocument.ComputeStatistics(2);tables=$taskDocument.Tables.Count;inline_images=$taskDocument.InlineShapes.Count;equations=$taskDocument.OMaths.Count;toc=$taskDocument.TablesOfContents.Count;pdf=$taskPdfPath} | ConvertTo-Json
}
finally {
    if ($null -ne $taskDocument) { $taskDocument.Close(0); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskDocument) | Out-Null }
    if ($null -ne $taskWord) {
        if ($taskOwnInstance -and $taskWord.Documents.Count -eq 0) { $taskWord.Quit(0) }
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord) | Out-Null
    }
}
