<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        // ── School structure ─────────────────────────────────────────────────

        Schema::create('schoolyear', function (Blueprint $table) {
            $table->increments('schoolyearid');
            $table->string('name', 100)->nullable();
        });

        Schema::create('class', function (Blueprint $table) {
            $table->increments('class_id');
            $table->string('name', 100)->nullable();
            $table->unsignedInteger('schoolyearid')->nullable();
        });

        Schema::create('section', function (Blueprint $table) {
            $table->increments('section_id');
            $table->unsignedInteger('class_id')->nullable();
            $table->string('name', 100)->nullable();
        });

        // ── Role profile tables ──────────────────────────────────────────────

        Schema::create('admin', function (Blueprint $table) {
            $table->increments('admin_id');
            $table->unsignedBigInteger('user_id')->nullable();
        });

        Schema::create('parent', function (Blueprint $table) {
            $table->increments('parent_id');
            $table->unsignedBigInteger('user_id')->nullable();
        });

        Schema::create('driver', function (Blueprint $table) {
            $table->increments('driver_id');
            $table->unsignedBigInteger('user_id')->nullable();
        });

        // ── Transport ────────────────────────────────────────────────────────

        Schema::create('bus', function (Blueprint $table) {
            $table->increments('bus_id');
            $table->string('plate_number', 50)->nullable();
        });

        Schema::create('route', function (Blueprint $table) {
            $table->increments('route_id');
            $table->string('name', 100)->nullable();
        });

        Schema::create('routestop', function (Blueprint $table) {
            $table->increments('stop_id');
            $table->unsignedInteger('route_id')->nullable();
            $table->string('name', 100)->nullable();
            $table->integer('stoporder')->nullable();
        });

        Schema::create('driverassignment', function (Blueprint $table) {
            $table->increments('driverassignment_id');
            $table->unsignedInteger('driver_id')->nullable();
            $table->unsignedInteger('bus_id')->nullable();
        });

        Schema::create('studentbusassignment', function (Blueprint $table) {
            $table->increments('sbassignment_id');
            $table->unsignedBigInteger('student_id')->nullable();
            $table->unsignedInteger('bus_id')->nullable();
            $table->unsignedInteger('route_id')->nullable();
            $table->unsignedInteger('stop_id')->nullable();
        });

        Schema::create('trip', function (Blueprint $table) {
            $table->increments('trip_id');
            $table->unsignedInteger('bus_id')->nullable();
            $table->unsignedInteger('driver_id')->nullable();
            $table->unsignedInteger('route_id')->nullable();
            $table->date('date')->nullable();
            $table->string('type', 50)->nullable();
        });

        Schema::create('trackingping', function (Blueprint $table) {
            $table->increments('ping_id');
            $table->unsignedInteger('trip_id')->nullable();
            $table->double('latitude')->nullable();
            $table->double('longitude')->nullable();
            $table->timestamp('capturedat')->nullable();
        });

        Schema::create('tripstopevent', function (Blueprint $table) {
            $table->increments('trpstopevent_id');
            $table->unsignedInteger('trip_id')->nullable();
            $table->unsignedInteger('stop_id')->nullable();
            $table->unsignedBigInteger('student_id')->nullable();
            $table->string('eventtype', 50)->nullable();
            $table->timestamp('eventat')->nullable();
        });

        // ── Academic ─────────────────────────────────────────────────────────

        Schema::create('enrollment', function (Blueprint $table) {
            $table->increments('enrollment_id');
            $table->unsignedBigInteger('student_id')->nullable();
            $table->unsignedInteger('section_id')->nullable();
            $table->string('status', 50)->nullable();
        });

        Schema::create('teacherassignment', function (Blueprint $table) {
            $table->increments('assignment_id');
            $table->unsignedBigInteger('teacher_id')->nullable();
            $table->unsignedInteger('section_id')->nullable();
            $table->unsignedBigInteger('subject_id')->nullable();
        });

        Schema::create('assessment', function (Blueprint $table) {
            $table->increments('assessment_id');
            $table->unsignedBigInteger('subject_id')->nullable();
            $table->unsignedInteger('section_id')->nullable();
            $table->string('title', 255)->nullable();
            $table->unsignedBigInteger('createdbyteacherid')->nullable();
            $table->string('assessmenttype', 50)->nullable();
            $table->date('date')->nullable();
            $table->double('maxscore')->nullable();
        });

        Schema::create('assessmentresult', function (Blueprint $table) {
            $table->increments('result_id');
            $table->unsignedInteger('assessment_id')->nullable();
            $table->unsignedBigInteger('student_id')->nullable();
            $table->double('score')->nullable();
            $table->string('grade', 10)->nullable();
            $table->timestamp('publishedat')->nullable();
        });

        Schema::create('homeworksubmission', function (Blueprint $table) {
            $table->increments('submission_id');
            $table->unsignedBigInteger('homework_id')->nullable();
            $table->unsignedBigInteger('student_id')->nullable();
            $table->timestamp('submittedat')->nullable();
            $table->double('score')->nullable();
            $table->string('status', 50)->nullable();
            // file_path added by: 2026_04_12_211156_add_file_path_to_homeworksubmission_table
        });

        // ── Attendance ───────────────────────────────────────────────────────

        Schema::create('attendancesession', function (Blueprint $table) {
            $table->increments('session_id');
            $table->unsignedInteger('section_id')->nullable();
            $table->date('date')->nullable();
        });

        Schema::create('studentattendance', function (Blueprint $table) {
            $table->increments('attendance_id');
            $table->unsignedInteger('session_id')->nullable();
            $table->unsignedBigInteger('student_id')->nullable();
            $table->string('status', 50)->nullable();
            $table->unsignedBigInteger('capturedbyuserid')->nullable();
        });

        Schema::create('teacherattendance', function (Blueprint $table) {
            $table->increments('hrattendance_id');
            $table->unsignedBigInteger('teacher_id')->nullable();
            $table->date('date')->nullable();
            $table->string('status', 50)->nullable();
            $table->unsignedBigInteger('capturedbyuserid')->nullable();
        });

        // ── Teacher HR ───────────────────────────────────────────────────────

        Schema::create('teacheravailability', function (Blueprint $table) {
            $table->increments('availability_id');
            $table->unsignedBigInteger('teacher_id')->nullable();
            $table->string('dayofweek', 20)->nullable();
            $table->time('start_time')->nullable();
            $table->time('end_time')->nullable();
            $table->string('availabilitytype', 50)->nullable();
        });

        Schema::create('vacationrequest', function (Blueprint $table) {
            $table->increments('vacation_id');
            $table->unsignedBigInteger('teacher_id')->nullable();
            $table->date('start_date')->nullable();
            $table->date('end_date')->nullable();
            $table->string('status', 50)->nullable();
            $table->unsignedInteger('approvedbyadmin_id')->nullable();
        });

        Schema::create('salarypayment', function (Blueprint $table) {
            $table->increments('salarypayment_id');
            $table->unsignedBigInteger('teacher_id')->nullable();
            $table->double('amount')->nullable();
            $table->string('periodmonth', 20)->nullable();
            $table->timestamp('paidat')->nullable();
        });

        // ── Schedule ─────────────────────────────────────────────────────────

        Schema::create('schedule', function (Blueprint $table) {
            $table->increments('schedule_id');
            $table->unsignedInteger('section_id')->nullable();
            $table->string('termname', 100)->nullable();
        });

        Schema::create('scheduleslot', function (Blueprint $table) {
            $table->increments('slot_id');
            $table->unsignedInteger('schedule_id')->nullable();
            $table->unsignedBigInteger('subject_id')->nullable();
            $table->unsignedBigInteger('teacher_id')->nullable();
            $table->string('dayofweek', 20)->nullable();
            $table->time('starttime')->nullable();
        });

        // ── Finance ──────────────────────────────────────────────────────────

        Schema::create('feeplan', function (Blueprint $table) {
            $table->increments('feeplan_id');
            $table->unsignedInteger('schoolyear_id')->nullable();
            $table->double('totalamount')->nullable();
            $table->string('name', 100)->nullable();
        });

        Schema::create('studentfeeplan', function (Blueprint $table) {
            $table->increments('account_id');
            $table->unsignedBigInteger('student_id')->nullable();
            $table->unsignedInteger('feeplan_id')->nullable();
            $table->double('balance')->nullable();
            // paid_amount, status, due_date, notes, timestamps added by:
            // 2026_05_12_133426_add_financial_fields_to_studentfeeplan_table
        });

        Schema::create('invoice', function (Blueprint $table) {
            $table->increments('invoice_id');
            $table->unsignedInteger('account_id')->nullable();
            $table->date('due_date')->nullable();
            $table->double('totalamount')->nullable();
            $table->string('status', 50)->nullable();
            // issued_date, notes added by: 2026_05_12_173223_add_issued_date_to_invoice_table
        });

        Schema::create('payment', function (Blueprint $table) {
            $table->increments('payment_id');
            $table->unsignedInteger('invoice_id')->nullable();
            $table->unsignedInteger('parent_id')->nullable();
            $table->double('amount')->nullable();
            $table->string('method', 50)->nullable();
            $table->timestamp('paidat')->nullable();
            // stripe_session_id, stripe_payment_intent, status added by:
            // 2026_04_13_180233_add_stripe_fields_to_payment_table
        });

        // ── Notifications ────────────────────────────────────────────────────

        Schema::create('notification', function (Blueprint $table) {
            $table->increments('notification_id');
            $table->string('title', 255)->nullable();
            $table->unsignedBigInteger('createdbyuserid')->nullable();
            $table->string('channel', 50)->nullable();
            $table->timestamp('created_at')->nullable();
            // body added by: 2026_05_11_163231_add_body_to_notification_table
        });

        Schema::create('notificationtrigger', function (Blueprint $table) {
            $table->increments('trigger_id');
            $table->string('triggertype', 50)->nullable();
            $table->text('rulejson')->nullable();
            $table->boolean('isactive')->nullable();
        });

        Schema::create('notificationrecipient', function (Blueprint $table) {
            $table->increments('recipient_id');
            $table->unsignedInteger('notification_id')->nullable();
            $table->unsignedBigInteger('user_id')->nullable();
            $table->string('status', 50)->nullable();
            $table->timestamp('deliveredat')->nullable();
            $table->timestamp('readat')->nullable();
        });

        Schema::create('notificationevent', function (Blueprint $table) {
            $table->increments('event_id');
            $table->unsignedInteger('trigger_id')->nullable();
            $table->string('relatedentitytype', 50)->nullable();
            $table->unsignedInteger('relatedentity_id')->nullable();
            $table->timestamp('createdat')->nullable();
        });

        // ── Guardians ────────────────────────────────────────────────────────

        Schema::create('studentguardian', function (Blueprint $table) {
            $table->increments('studentguardian_id');
            $table->unsignedBigInteger('student_id')->nullable();
            $table->unsignedInteger('parent_id')->nullable();
            $table->string('relationship', 50)->nullable();
            $table->boolean('isprimary')->nullable();
        });

        // ── Surveillance ─────────────────────────────────────────────────────

        Schema::create('camera', function (Blueprint $table) {
            $table->increments('camera_id');
            $table->string('location', 255)->nullable();
            $table->boolean('isactive')->nullable();
            // code, stream_url, stream_id added by: 2026_05_12_000001_add_fight_detection_fields
        });

        Schema::create('surveillanceevent', function (Blueprint $table) {
            $table->increments('survevent_id');
            $table->unsignedInteger('camera_id')->nullable();
            $table->string('detectedtype', 50)->nullable();
            $table->timestamp('detectedat')->nullable();
            $table->string('severity', 50)->nullable();
            $table->unsignedBigInteger('relatedstudent_id')->nullable();
            $table->unsignedInteger('relatedsection_id')->nullable();
            $table->unsignedInteger('relatedassessment_id')->nullable();
            // confidence, footage_path, status added by: 2026_05_12_000001_add_fight_detection_fields
        });

        // ── Analytics ────────────────────────────────────────────────────────

        Schema::create('analyticsreport', function (Blueprint $table) {
            $table->increments('report_id');
            $table->string('reporttype', 50)->nullable();
            $table->date('periodstart')->nullable();
            $table->date('periodend')->nullable();
            $table->timestamp('generated_at')->nullable();
            $table->unsignedInteger('generatedbyadmin_id')->nullable();
        });

        Schema::create('analyticsmetric', function (Blueprint $table) {
            $table->increments('metric_id');
            $table->unsignedInteger('report_id')->nullable();
            $table->string('metricname', 100)->nullable();
            $table->string('metricvalue', 100)->nullable();
            $table->string('dimension', 100)->nullable();
        });
    }

    public function down(): void
    {
        // Drop in reverse dependency order
        Schema::dropIfExists('analyticsmetric');
        Schema::dropIfExists('analyticsreport');
        Schema::dropIfExists('surveillanceevent');
        Schema::dropIfExists('camera');
        Schema::dropIfExists('studentguardian');
        Schema::dropIfExists('notificationevent');
        Schema::dropIfExists('notificationrecipient');
        Schema::dropIfExists('notificationtrigger');
        Schema::dropIfExists('notification');
        Schema::dropIfExists('payment');
        Schema::dropIfExists('invoice');
        Schema::dropIfExists('studentfeeplan');
        Schema::dropIfExists('feeplan');
        Schema::dropIfExists('scheduleslot');
        Schema::dropIfExists('schedule');
        Schema::dropIfExists('salarypayment');
        Schema::dropIfExists('vacationrequest');
        Schema::dropIfExists('teacheravailability');
        Schema::dropIfExists('teacherattendance');
        Schema::dropIfExists('studentattendance');
        Schema::dropIfExists('attendancesession');
        Schema::dropIfExists('homeworksubmission');
        Schema::dropIfExists('assessmentresult');
        Schema::dropIfExists('assessment');
        Schema::dropIfExists('teacherassignment');
        Schema::dropIfExists('enrollment');
        Schema::dropIfExists('tripstopevent');
        Schema::dropIfExists('trackingping');
        Schema::dropIfExists('trip');
        Schema::dropIfExists('studentbusassignment');
        Schema::dropIfExists('driverassignment');
        Schema::dropIfExists('routestop');
        Schema::dropIfExists('route');
        Schema::dropIfExists('bus');
        Schema::dropIfExists('driver');
        Schema::dropIfExists('parent');
        Schema::dropIfExists('admin');
        Schema::dropIfExists('section');
        Schema::dropIfExists('class');
        Schema::dropIfExists('schoolyear');
    }
};
