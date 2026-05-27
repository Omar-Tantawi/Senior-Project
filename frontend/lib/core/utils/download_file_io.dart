// Mobile / desktop implementation — writes bytes to disk and opens the file.
import 'dart:io';

import 'package:open_file/open_file.dart';
import 'package:path_provider/path_provider.dart';

Future<void> downloadFile(List<int> bytes, String filename) async {
  final dir  = Platform.isAndroid
      ? Directory('/storage/emulated/0/Download')
      : await getApplicationDocumentsDirectory();
  final file = File('${dir.path}/$filename');
  await file.writeAsBytes(bytes);
  await OpenFile.open(file.path);
}
