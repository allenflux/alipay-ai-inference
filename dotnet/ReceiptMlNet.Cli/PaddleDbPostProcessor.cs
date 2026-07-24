using Clipper2Lib;
using OpenCvSharp;

/// <summary>Frozen DB post-processing settings from PaddleOCR v2.</summary>
internal sealed record PaddleDbOptions(
    float BinaryThreshold,
    float BoxThreshold,
    float UnclipRatio,
    bool UseDilation,
    string ScoreMode);

/// <summary>
/// PaddleOCR DB text-detection post-process. This deliberately keeps the
/// contour, mini-box, score and unclip stages explicit rather than replacing
/// them with a detector-specific heuristic.
/// </summary>
internal static class PaddleDbPostProcessor
{
    private const int MaxCandidates = 1000;

    public static IReadOnlyList<Point2f[]> Process(
        float[] probabilityValues,
        int mapHeight,
        int mapWidth,
        int sourceHeight,
        int sourceWidth,
        PaddleDbOptions options)
    {
        if (mapHeight <= 0 || mapWidth <= 0 || probabilityValues.Length < mapHeight * mapWidth)
        {
            throw new InvalidOperationException("Paddle DB ONNX output has an invalid probability-map shape");
        }
        if (options.ScoreMode is not ("fast" or "slow"))
        {
            throw new InvalidOperationException($"Unsupported PaddleOCR det_db_score_mode: {options.ScoreMode}");
        }

        using var probability = new Mat(mapHeight, mapWidth, MatType.CV_32FC1);
        System.Runtime.InteropServices.Marshal.Copy(probabilityValues, 0, probability.Data, mapHeight * mapWidth);
        using var thresholdedFloat = new Mat();
        Cv2.Threshold(probability, thresholdedFloat, options.BinaryThreshold, 255.0, ThresholdTypes.Binary);
        using var thresholded = new Mat();
        thresholdedFloat.ConvertTo(thresholded, MatType.CV_8UC1);

        Mat contourInput = thresholded;
        Mat? dilated = null;
        try
        {
            if (options.UseDilation)
            {
                using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(2, 2));
                dilated = new Mat();
                Cv2.Dilate(thresholded, dilated, kernel);
                contourInput = dilated;
            }

            Cv2.FindContours(contourInput, out var contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
            var boxes = new List<Point2f[]>();
            foreach (var contour in contours.Take(MaxCandidates))
            {
                var (box, shortSide) = GetMiniBox(contour);
                if (shortSide < 3.0f)
                {
                    continue;
                }
                var score = string.Equals(options.ScoreMode, "fast", StringComparison.Ordinal)
                    ? BoxScoreFast(probability, box)
                    : BoxScoreSlow(probability, contour);
                if (score < options.BoxThreshold)
                {
                    continue;
                }

                var expanded = Unclip(box, options.UnclipRatio);
                if (expanded is null)
                {
                    continue;
                }
                var (expandedBox, expandedShortSide) = GetMiniBox(expanded);
                if (expandedShortSide < 5.0f)
                {
                    continue;
                }

                // DBPostProcess deliberately permits the last coordinate to
                // equal dest_width/dest_height here.  Paddle's following
                // TextDetector.filter_tag_det_res stage performs the final
                // clockwise ordering, integer clipping to width - 1 / height
                // - 1, and its small-box rejection.
                var scaled = expandedBox
                    .Select(point => new Point2f(
                        ClampRound(point.X / mapWidth * sourceWidth, 0, sourceWidth),
                        ClampRound(point.Y / mapHeight * sourceHeight, 0, sourceHeight)))
                    .ToArray();
                var accepted = FilterTagDetectionResult(scaled, sourceHeight, sourceWidth);
                if (accepted is not null)
                {
                    boxes.Add(accepted);
                }
            }
            return SortBoxes(boxes);
        }
        finally
        {
            dilated?.Dispose();
        }
    }

    private static float BoxScoreFast(Mat probability, Point2f[] box)
    {
        var xmin = Math.Clamp((int)Math.Floor(box.Min(point => point.X)), 0, probability.Cols - 1);
        var xmax = Math.Clamp((int)Math.Ceiling(box.Max(point => point.X)), 0, probability.Cols - 1);
        var ymin = Math.Clamp((int)Math.Floor(box.Min(point => point.Y)), 0, probability.Rows - 1);
        var ymax = Math.Clamp((int)Math.Ceiling(box.Max(point => point.Y)), 0, probability.Rows - 1);
        if (xmax < xmin || ymax < ymin)
        {
            return 0.0f;
        }

        using var mask = new Mat(ymax - ymin + 1, xmax - xmin + 1, MatType.CV_8UC1, Scalar.All(0));
        var local = box
            .Select(point => new Point(
                Math.Clamp((int)(point.X - xmin), 0, mask.Cols - 1),
                Math.Clamp((int)(point.Y - ymin), 0, mask.Rows - 1)))
            .ToArray();
        Cv2.FillPoly(mask, new[] { local }, Scalar.All(1));
        using var scoreRegion = new Mat(probability, new Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
        return (float)Cv2.Mean(scoreRegion, mask).Val0;
    }

    private static float BoxScoreSlow(Mat probability, Point[] contour)
    {
        var xmin = Math.Clamp(contour.Min(point => point.X), 0, probability.Cols - 1);
        var xmax = Math.Clamp(contour.Max(point => point.X), 0, probability.Cols - 1);
        var ymin = Math.Clamp(contour.Min(point => point.Y), 0, probability.Rows - 1);
        var ymax = Math.Clamp(contour.Max(point => point.Y), 0, probability.Rows - 1);
        if (xmax < xmin || ymax < ymin)
        {
            return 0.0f;
        }

        using var mask = new Mat(ymax - ymin + 1, xmax - xmin + 1, MatType.CV_8UC1, Scalar.All(0));
        var local = contour
            .Select(point => new Point(
                Math.Clamp(point.X - xmin, 0, mask.Cols - 1),
                Math.Clamp(point.Y - ymin, 0, mask.Rows - 1)))
            .ToArray();
        Cv2.FillPoly(mask, new[] { local }, Scalar.All(1));
        using var scoreRegion = new Mat(probability, new Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
        return (float)Cv2.Mean(scoreRegion, mask).Val0;
    }

    private static Point[]? Unclip(Point2f[] box, float unclipRatio)
    {
        var area = Math.Abs(PolygonArea(box));
        var perimeter = PolygonPerimeter(box);
        if (area <= 0.0 || perimeter <= 0.0)
        {
            return null;
        }
        var distance = area * unclipRatio / perimeter;
        var path = new Path64();
        foreach (var point in box)
        {
            // pyclipper's integer coordinate conversion truncates positive
            // mini-box coordinates, so keep that behaviour here.
            path.Add(new Point64((long)point.X, (long)point.Y));
        }
        var offset = new ClipperOffset();
        offset.AddPath(path, JoinType.Round, EndType.Polygon);
        var expanded = new Paths64();
        offset.Execute(distance, expanded);
        if (expanded.Count != 1 || expanded[0].Count < 3)
        {
            return null;
        }
        return expanded[0].Select(point => new Point((int)point.X, (int)point.Y)).ToArray();
    }

    private static (Point2f[] Box, float ShortSide) GetMiniBox(IEnumerable<Point> contour)
    {
        var contourArray = contour.ToArray();
        if (contourArray.Length < 3)
        {
            return (Array.Empty<Point2f>(), 0.0f);
        }
        var rectangle = Cv2.MinAreaRect(contourArray);
        return (OrderPointsClockwise(Cv2.BoxPoints(rectangle)), Math.Min(rectangle.Size.Width, rectangle.Size.Height));
    }

    private static Point2f[] OrderPointsClockwise(IReadOnlyList<Point2f> points)
    {
        if (points.Count != 4)
        {
            return points.ToArray();
        }
        var sorted = points.OrderBy(point => point.X).ToArray();
        var first = sorted[1].Y > sorted[0].Y ? 0 : 1;
        var second = 1 - first;
        var third = sorted[3].Y > sorted[2].Y ? 2 : 3;
        var fourth = third == 2 ? 3 : 2;
        return new[] { sorted[first], sorted[third], sorted[fourth], sorted[second] };
    }

    /// <summary>
    /// Exact final validation stage used by PaddleOCR's TextDetector after DB
    /// post-processing.  Keeping this separate is important: DB itself clips
    /// to dest_width/dest_height, while this stage clips to valid pixel
    /// coordinates (width - 1 / height - 1) before rejecting tiny boxes.
    /// </summary>
    private static Point2f[]? FilterTagDetectionResult(
        IReadOnlyList<Point2f> points,
        int imageHeight,
        int imageWidth)
    {
        if (imageHeight <= 0 || imageWidth <= 0)
        {
            return null;
        }

        var ordered = OrderDetectionPointsClockwise(points);
        for (var index = 0; index < ordered.Length; index++)
        {
            // Paddle assigns into an int32 array here, which truncates these
            // already rounded DB coordinates before clipping.
            ordered[index] = new Point2f(
                Math.Clamp((int)ordered[index].X, 0, imageWidth - 1),
                Math.Clamp((int)ordered[index].Y, 0, imageHeight - 1));
        }

        if (ordered.Length != 4)
        {
            return null;
        }
        var width = (int)Distance(ordered[0], ordered[1]);
        var height = (int)Distance(ordered[0], ordered[3]);
        return width <= 3 || height <= 3 ? null : ordered;
    }

    /// <summary>
    /// Matches <c>TextDetector.order_points_clockwise</c>, which is distinct
    /// from DBPostProcess.get_mini_boxes: Paddle chooses top-left / bottom-right
    /// with x+y, then top-right / bottom-left with y-x.  A pure x-sort changes
    /// the crop order for strongly skewed text quadrilaterals.
    /// </summary>
    private static Point2f[] OrderDetectionPointsClockwise(IReadOnlyList<Point2f> points)
    {
        if (points.Count != 4)
        {
            return points.ToArray();
        }

        var topLeft = 0;
        var bottomRight = 0;
        for (var index = 1; index < points.Count; index++)
        {
            var sum = points[index].X + points[index].Y;
            var currentTopLeftSum = points[topLeft].X + points[topLeft].Y;
            var currentBottomRightSum = points[bottomRight].X + points[bottomRight].Y;
            // NumPy argmin/argmax keep the first tied element, so use strict
            // comparisons rather than <= or >=.
            if (sum < currentTopLeftSum)
            {
                topLeft = index;
            }
            if (sum > currentBottomRightSum)
            {
                bottomRight = index;
            }
        }

        var remaining = Enumerable.Range(0, points.Count)
            .Where(index => index != topLeft && index != bottomRight)
            .ToArray();
        if (remaining.Length != 2)
        {
            return points.ToArray();
        }
        var firstDifference = points[remaining[0]].Y - points[remaining[0]].X;
        var secondDifference = points[remaining[1]].Y - points[remaining[1]].X;
        var topRight = firstDifference <= secondDifference ? remaining[0] : remaining[1];
        var bottomLeft = topRight == remaining[0] ? remaining[1] : remaining[0];
        return new[] { points[topLeft], points[topRight], points[bottomRight], points[bottomLeft] };
    }

    private static IReadOnlyList<Point2f[]> SortBoxes(List<Point2f[]> boxes)
    {
        // LINQ's ordered enumeration is stable, matching Python's sorted()
        // when two detections have identical top-left coordinates.
        var stableOrder = boxes
            .Select((box, index) => (Box: box, Index: index))
            .OrderBy(item => item.Box[0].Y)
            .ThenBy(item => item.Box[0].X)
            .ThenBy(item => item.Index)
            .Select(item => item.Box)
            .ToList();
        boxes.Clear();
        boxes.AddRange(stableOrder);
        // This is Paddle's second ordering pass for text boxes sharing a line.
        for (var index = 0; index < boxes.Count - 1; index++)
        {
            for (var cursor = index; cursor >= 0; cursor--)
            {
                var current = boxes[cursor];
                var following = boxes[cursor + 1];
                if (Math.Abs(following[0].Y - current[0].Y) < 10.0f && following[0].X < current[0].X)
                {
                    boxes[cursor] = following;
                    boxes[cursor + 1] = current;
                    continue;
                }
                break;
            }
        }
        return boxes;
    }

    private static float ClampRound(float value, int minimum, int maximum)
    {
        return Math.Clamp((float)Math.Round(value, MidpointRounding.ToEven), minimum, maximum);
    }

    private static double PolygonArea(IReadOnlyList<Point2f> points)
    {
        var area = 0.0;
        for (var index = 0; index < points.Count; index++)
        {
            var next = points[(index + 1) % points.Count];
            area += points[index].X * next.Y - next.X * points[index].Y;
        }
        return area / 2.0;
    }

    private static double PolygonPerimeter(IReadOnlyList<Point2f> points)
    {
        var length = 0.0;
        for (var index = 0; index < points.Count; index++)
        {
            var next = points[(index + 1) % points.Count];
            var dx = points[index].X - next.X;
            var dy = points[index].Y - next.Y;
            length += Math.Sqrt(dx * dx + dy * dy);
        }
        return length;
    }

    private static float Distance(Point2f first, Point2f second)
    {
        var dx = first.X - second.X;
        var dy = first.Y - second.Y;
        return MathF.Sqrt(dx * dx + dy * dy);
    }
}
