<?php

namespace App\Http\Requests\Attendance;

use Illuminate\Foundation\Http\FormRequest;

class StoreAttendanceRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'section_id'           => 'required|integer',
            'date'                 => 'required|date',
            'records'              => 'required|array|min:1',
            'records.*.student_id' => 'required|integer',
            'records.*.status'     => 'required|in:present,absent,late,excused',
        ];
    }
}
